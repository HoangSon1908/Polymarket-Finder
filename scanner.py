from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import streamlit as st

st.set_page_config(
    page_title="Polymarket Pro Wallet Finder — 4-Pillar Pipeline (v5.6)",
    page_icon="📡",
    layout="wide",
)

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & HẰNG SỐ API (DATA API V2)
# ==============================================================================

LEADERBOARD_V2_URL = "https://data-api.polymarket.com/v2/leaderboard"
LEADERBOARD_V1_URL = "https://data-api.polymarket.com/v1/leaderboard"
POSITIONS_V2_URL = "https://data-api.polymarket.com/v2/positions"
ACTIVITY_V2_URL = "https://data-api.polymarket.com/v2/activity"

VN_TZ = timezone(timedelta(hours=7))
EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_THREAD_LOCAL = threading.local()

CATEGORY_OPTIONS: dict[str, str] = {
    "Sports (NBA, Tennis, Bóng đá, UFC...)": "SPORTS",
    "Esports (CS2, Dota 2, LoL, Valorant...)": "ESPORTS",
    "Finance (Chỉ số, Cổ phiếu, Vàng, Hàng hóa)": "FINANCE",
    "Economics (Lãi suất Fed, CPI, GDP, Vĩ mô)": "ECONOMICS",
    "Politics (Chính trị & Bầu cử)": "POLITICS",
    "Crypto (Tiền mã hóa)": "CRYPTO",
    "Weather (Thời tiết & Nhiệt độ)": "WEATHER",
    "Culture (Văn hóa & Giải trí)": "CULTURE",
    "Mentions (Từ khóa & Phát ngôn)": "MENTIONS",
    "Tech (Công nghệ & AI)": "TECH",
    "All Categories (Toàn sàn Polymarket)": "OVERALL",
}

RANGE_OPTIONS: dict[str, str] = {
    "Today (24h qua)": "day",
    "Weekly (7 ngày qua)": "week",
    "Monthly (30 ngày qua)": "month",
    "All (Toàn thời gian)": "all",
}


def get_http_session() -> requests.Session:
    """Mỗi worker thread giữ riêng 1 Session Keep-Alive để tái sử dụng kết nối TLS."""
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Polymarket-Wallet-Finder/5.6-4Pillars",
            }
        )
        _THREAD_LOCAL.session = sess
    return sess


def request_json_with_retry(
    url: str,
    params: dict[str, Any],
    retries: int = 3,
    timeout: int = 12,
) -> tuple[Any, int]:
    """Trả về (json_payload, http_status_code) với cơ chế tự thử lại khi gặp 429/lỗi mạng."""
    session = get_http_session()
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(0.8 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json(), resp.status_code
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(f"Gọi {url} thất bại (params={params}): {last_err}")


def _extract_wallet_address(item: dict[str, Any]) -> str:
    candidate_keys = (
        "proxy_wallet",
        "proxyWallet",
        "wallet",
        "wallet_address",
        "walletAddress",
        "user_address",
        "userAddress",
        "address",
        "account",
        "user",
        "id",
    )
    for k in candidate_keys:
        val = item.get(k)
        if isinstance(val, str) and EVM_ADDR_RE.match(val.strip()):
            return val.strip().lower()
        elif isinstance(val, dict):
            nested = _extract_wallet_address(val)
            if nested:
                return nested

    for val in item.values():
        if isinstance(val, str) and EVM_ADDR_RE.match(val.strip()):
            return val.strip().lower()
        elif isinstance(val, dict):
            for sub_val in val.values():
                if isinstance(sub_val, str) and EVM_ADDR_RE.match(sub_val.strip()):
                    return sub_val.strip().lower()
    return ""


def _pick_first_str(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        val = item.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for sub_key in ("user", "profile", "account"):
        sub = item.get(sub_key)
        if isinstance(sub, dict):
            for k in keys:
                val = sub.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


def _pick_first_float(item: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for k in keys:
        val = item.get(k)
        if val is not None and not isinstance(val, (dict, list, bool)):
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return default


def _parse_timestamp_utc(raw_ts: Any) -> datetime | None:
    if raw_ts is None:
        return None
    if isinstance(raw_ts, (int, float)):
        ts_val = float(raw_ts)
        if ts_val <= 0:
            return None
        if ts_val > 1e12:
            ts_val /= 1000.0
        try:
            return datetime.fromtimestamp(ts_val, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(raw_ts, str):
        s = raw_ts.strip()
        if not s:
            return None
        if s.isdigit():
            return _parse_timestamp_utc(float(s))
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _to_unix_ts(raw_ts: Any) -> float:
    dt = _parse_timestamp_utc(raw_ts)
    return dt.timestamp() if dt is not None else float("nan")


def _build_wallet_csv_name(user_name: str, wallet: str) -> str:
    raw = (user_name or "").strip().lstrip("@")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_").lower()
    short_w = wallet[:8].lower() if wallet else "wallet"
    if slug and not slug.startswith("0x") and not slug.startswith("direct"):
        return f"{slug}_{short_w}.csv"
    return f"{short_w}.csv"


def _extract_batch_and_cursor(payload: Any) -> tuple[list[dict[str, Any]], str | None, bool]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)], None, False
    if isinstance(payload, dict):
        batch = (
            payload.get("data")
            or payload.get("results")
            or payload.get("leaderboard")
            or payload.get("items")
            or []
        )
        pagination = payload.get("pagination") or {}
        next_cursor = (
            pagination.get("next_cursor")
            or payload.get("next_cursor")
            or payload.get("cursor")
        )
        has_more = bool(pagination.get("has_more", bool(next_cursor)))
        return [r for r in batch if isinstance(r, dict)], next_cursor, has_more
    return [], None, False


def _fetch_positions_by_status(
    wallet: str,
    status_val: str,
    max_pages: int = 8,
    stop_cid_cap: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    unique_cids: set[str] = set()
    cursor: str | None = None
    seen_cursors: set[str] = set()
    offset = 0
    capped = False

    for _ in range(max_pages):
        params: dict[str, Any] = {"user": wallet, "status": status_val, "limit": 500}
        if cursor:
            params["cursor"] = cursor
        elif offset > 0:
            params["offset"] = offset

        try:
            payload, _ = request_json_with_retry(POSITIONS_V2_URL, params=params, retries=2, timeout=10)
        except Exception:
            break

        batch, next_cursor, has_more = _extract_batch_and_cursor(payload)
        if not batch:
            break

        for r in batch:
            r["_queried_status"] = status_val
            rows.append(r)
            cid = str(r.get("condition_id") or r.get("conditionId") or "").strip()
            if cid:
                unique_cids.add(cid)

        if stop_cid_cap is not None and len(unique_cids) > stop_cid_cap:
            capped = True
            break

        if next_cursor and has_more:
            if next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        elif len(batch) >= 500:
            cursor = None
            offset += len(batch)
        else:
            break

    return rows, capped


def _fetch_activities_v2(wallet: str, max_pages: int = 5) -> list[dict[str, Any]]:
    """Kéo lịch sử Activity hỗ trợ đầy đủ cả phân trang cursor lẫn offset fallback."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    seen_uids: set[str] = set()
    offset = 0

    for _ in range(max_pages):
        params: dict[str, Any] = {"user": wallet, "limit": 500}
        if cursor:
            params["cursor"] = cursor
        elif offset > 0:
            params["offset"] = offset

        try:
            payload, _ = request_json_with_retry(ACTIVITY_V2_URL, params=params, retries=2, timeout=10)
        except Exception:
            break

        batch, next_cursor, has_more = _extract_batch_and_cursor(payload)
        if not batch:
            break

        added = 0
        for r in batch:
            uid = str(
                r.get("id")
                or r.get("transaction_hash")
                or r.get("transactionHash")
                or ""
            ).strip()
            if uid:
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
            rows.append(r)
            added += 1

        if added == 0:
            break

        if next_cursor and has_more:
            if next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        elif len(batch) >= 500:
            cursor = None
            offset += len(batch)
        else:
            break

    return rows


# ==============================================================================
# BƯỚC 1: THU THẬP TOP N VÍ (CHIA LÔ AN TOÀN + SOI TRỰC TIẾP 0x...)
# ==============================================================================

def step1_fetch_leaderboard(
    category: str,
    time_period: str,
    order_by: str = "PNL",
    top_n: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cat_code = category.strip().upper()
    tp_code = time_period.strip().lower()
    ob_code = order_by.strip().upper()

    raw_records: list[dict[str, Any]] = []
    seen_wallets_raw: set[str] = set()
    used_endpoint = LEADERBOARD_V2_URL
    errors: list[str] = []
    debug_notes: list[str] = []

    for page_limit in (min(20, top_n), 10):
        raw_records.clear()
        seen_wallets_raw.clear()
        cursor: str | None = None
        offset = 0
        max_pages = max(1, math.ceil(top_n / page_limit)) + 1

        for page_idx in range(max_pages):
            if len(raw_records) >= top_n:
                break

            req_limit = min(page_limit, top_n - len(raw_records))
            if cursor is not None:
                params: dict[str, Any] = {"cursor": cursor}
            else:
                params = {
                    "category": cat_code,
                    "time_period": tp_code,
                    "order_by": ob_code,
                    "limit": req_limit,
                }
                if offset > 0:
                    params["offset"] = offset

            try:
                payload, http_code = request_json_with_retry(LEADERBOARD_V2_URL, params=params)
            except Exception as exc:
                if cursor is not None:
                    try:
                        retry_params = {
                            "category": cat_code,
                            "time_period": tp_code,
                            "order_by": ob_code,
                            "limit": req_limit,
                            "cursor": cursor,
                        }
                        payload, http_code = request_json_with_retry(LEADERBOARD_V2_URL, params=retry_params)
                    except Exception as exc2:
                        errors.append(f"[v2 cursor] {exc2}")
                        break
                else:
                    errors.append(f"[v2 p{page_idx}] {exc}")
                    break

            batch, next_cursor, has_more = _extract_batch_and_cursor(payload)
            if not batch:
                preview = str(payload)[:260]
                debug_notes.append(f"HTTP {http_code} với limit={req_limit} trả batch rỗng: {preview}")
                break

            added_new = 0
            for r in batch:
                w = _extract_wallet_address(r)
                k = w if w else f"row_{len(raw_records)}_{added_new}"
                if k in seen_wallets_raw:
                    continue
                seen_wallets_raw.add(k)
                raw_records.append(r)
                added_new += 1

            if added_new == 0:
                break

            if next_cursor and has_more:
                cursor = next_cursor
            elif len(batch) >= req_limit and len(raw_records) < top_n:
                cursor = None
                offset += len(batch)
            else:
                break

        if raw_records:
            break

    if not raw_records:
        used_endpoint = LEADERBOARD_V1_URL
        offset = 0
        while len(raw_records) < top_n:
            batch_limit = min(20, top_n - len(raw_records))
            try:
                payload_v1, _ = request_json_with_retry(
                    LEADERBOARD_V1_URL,
                    params={
                        "category": cat_code,
                        "timePeriod": tp_code.upper(),
                        "orderBy": ob_code,
                        "limit": batch_limit,
                        "offset": offset,
                    },
                    retries=2,
                )
                batch_v1, _, _ = _extract_batch_and_cursor(payload_v1)
                if not batch_v1:
                    break
                raw_records.extend(batch_v1)
                offset += len(batch_v1)
                if len(batch_v1) < batch_limit:
                    break
            except Exception as exc_v1:
                errors.append(f"[v1] {exc_v1}")
                break

    raw_records = raw_records[:top_n]

    meta: dict[str, Any] = {
        "fetched_at_vn": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S GMT+7"),
        "endpoint": used_endpoint,
        "category": cat_code,
        "time_period": tp_code,
        "order_by": ob_code,
        "requested_top_n": top_n,
        "returned_count": 0,
        "errors": errors,
        "debug_notes": debug_notes,
        "raw_records": raw_records,
    }

    if not raw_records:
        return pd.DataFrame(), meta

    normalized_rows: list[dict[str, Any]] = []
    seen_wallets: set[str] = set()

    for idx, item in enumerate(raw_records, start=1):
        wallet = _extract_wallet_address(item)
        dedup_key = wallet if wallet else f"unknown_{idx}"
        if dedup_key in seen_wallets:
            continue
        seen_wallets.add(dedup_key)

        raw_rank = item.get("rank") or item.get("position")
        try:
            rank_val = int(raw_rank) if raw_rank is not None else idx
        except (ValueError, TypeError):
            rank_val = idx

        user_name = _pick_first_str(
            item, ("user_name", "userName", "username", "name", "pseudonym", "display_name")
        )
        if not user_name:
            user_name = f"{wallet[:6]}...{wallet[-4:]}" if wallet else f"Trader #{rank_val}"

        pnl_val = _pick_first_float(item, ("pnl", "total_pnl", "totalPnl", "profit", "net_pnl"))
        vol_val = _pick_first_float(item, ("vol", "volume", "total_volume", "totalVolume", "usdc_volume"))
        pnl_vol_pct = (pnl_val / vol_val * 100.0) if vol_val > 0 else 0.0

        normalized_rows.append(
            {
                "rank": rank_val,
                "user_name": user_name,
                "wallet": wallet if wallet else "",
                "pnl_usdc": round(pnl_val, 2),
                "ui_volume_usdc": round(vol_val, 2),
                "pnl_to_ui_vol_pct": round(pnl_vol_pct, 2),
                "x_username": _pick_first_str(item, ("x_username", "xUsername", "twitter", "x_handle")),
                "verified": bool(item.get("verified_badge") or item.get("verifiedBadge") or item.get("verified") or False),
                "category": cat_code,
                "time_period": tp_code,
                "profile_url": f"https://polymarket.com/profile/{wallet}" if wallet else "",
            }
        )

    if not normalized_rows:
        return pd.DataFrame(), meta

    df = pd.DataFrame(normalized_rows).sort_values("rank", ascending=True).reset_index(drop=True)
    meta["returned_count"] = len(df)
    return df, meta


# ==============================================================================
# BƯỚC 2: THĂM DÒ + LƯU CACHE POSITIONS ĐỂ BƯỚC 3 TÁI SỬ DỤNG 100%
# ==============================================================================

def _probe_and_cache_positions(wallet: str, max_settled_cap: int, max_dead_cap: int) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    cutoff_30d = now_utc - timedelta(days=30)

    days_since_last: float | None = None
    recent_market_title = ""
    recent_30d_cids: set[str] = set()
    seed_acts: list[dict[str, Any]] = []

    try:
        act_payload, _ = request_json_with_retry(
            ACTIVITY_V2_URL, params={"user": wallet, "limit": 100}, retries=2, timeout=10
        )
        seed_acts, _, _ = _extract_batch_and_cursor(act_payload)
        latest_dt: datetime | None = None
        for r in seed_acts:
            if not recent_market_title:
                recent_market_title = str(r.get("title") or r.get("question") or "").strip()
            dt = _parse_timestamp_utc(r.get("timestamp") or r.get("created_at") or r.get("time"))
            if dt:
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
                if dt >= cutoff_30d:
                    cid = str(r.get("condition_id") or r.get("conditionId") or r.get("market") or "").strip()
                    if cid:
                        recent_30d_cids.add(cid)
        if latest_dt is not None:
            days_since_last = max(0.0, (now_utc - latest_dt).total_seconds() / 86400.0)
    except Exception:
        pass

    open_pages = max(2, math.ceil((max_dead_cap + 200) / 500))
    open_rows, open_capped = _fetch_positions_by_status(
        wallet, "OPEN", max_pages=open_pages, stop_cid_cap=max_dead_cap + 100
    )
    live_open_cnt = 0
    dead_zero_open_cnt = 0

    for r in open_rows:
        cur_p = _pick_first_float(r, ("current_price", "curPrice", "price"), 0.0)
        cur_v = _pick_first_float(r, ("current_value", "currentValue"), 0.0)
        cur_s = _pick_first_float(r, ("current_size", "size"), 0.0)
        if not recent_market_title:
            recent_market_title = str(r.get("title") or "").strip()

        if cur_p <= 0.01 and cur_v <= 0.5:
            dead_zero_open_cnt += 1
        elif 0.02 <= cur_p <= 0.98 and (cur_v > 1.0 or cur_s > 1.0):
            live_open_cnt += 1
        elif cur_p > 0.98 and cur_v > 1.0:
            pass
        else:
            dead_zero_open_cnt += 1

    settled_cids: set[str] = set()
    max_pages_closed = min(10, max(2, math.ceil((max_settled_cap + 200) / 500)))
    closed_rows, settled_capped = _fetch_positions_by_status(
        wallet, "CLOSED", max_pages=max_pages_closed, stop_cid_cap=max_settled_cap
    )
    for r in closed_rows:
        cid = str(r.get("condition_id") or r.get("conditionId") or "").strip()
        if cid:
            settled_cids.add(cid)
        if not recent_market_title:
            recent_market_title = str(r.get("title") or "").strip()

    redeem_rows: list[dict[str, Any]] = []
    lost_rows: list[dict[str, Any]] = []
    if not settled_capped:
        redeem_rows, _ = _fetch_positions_by_status(wallet, "REDEEMABLE", max_pages=2, stop_cid_cap=500)
        lost_rows, _ = _fetch_positions_by_status(wallet, "REDEEMABLE_LOST", max_pages=2, stop_cid_cap=500)
        for r in redeem_rows + lost_rows:
            cid = str(r.get("condition_id") or r.get("conditionId") or "").strip()
            if cid:
                settled_cids.add(cid)

    settled_count = len(settled_cids)
    if settled_count > max_settled_cap:
        settled_capped = True

    all_cached_positions = open_rows + closed_rows + redeem_rows + lost_rows

    return {
        "settled_markets": settled_count,
        "settled_capped": settled_capped,
        "live_open": live_open_cnt,
        "dead_zero_open": dead_zero_open_cnt,
        "open_capped": open_capped,
        "recent_30d_mkts": len(recent_30d_cids),
        "days_since_last": round(days_since_last, 1) if days_since_last is not None else None,
        "recent_market": recent_market_title[:75] if recent_market_title else "N/A",
        "raw_positions": all_cached_positions,
        "seed_acts": seed_acts,
    }


def step2_surface_filter(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    wallet_cache: dict[str, dict[str, Any]],
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t0 = time.perf_counter()
    if df.empty:
        return df.copy(), df.copy(), {"elapsed_s": 0.0, "input_count": 0, "pre_pass_count": 0, "passed_count": 0}

    work_df = df.copy()
    min_pnl = float(cfg["min_net_pnl"])
    min_pv = float(cfg["min_pv_pct"])
    min_mkts = int(cfg["min_markets"])
    max_mkts = int(cfg["max_markets"])
    max_inactive_days = float(cfg["max_inactive_days"])
    max_dead_open = int(cfg["max_dead_open"])

    pre_pass_wallets: list[str] = []
    pre_fail_reasons: dict[str, list[str]] = {}

    for _, row in work_df.iterrows():
        w = str(row["wallet"]).strip().lower()
        is_direct = bool(row.get("is_direct_lookup", False))
        reasons = []
        if not w or not w.startswith("0x"):
            reasons.append("Thiếu địa chỉ ví 0x")
        if not is_direct:
            if float(row["pnl_usdc"]) < min_pnl:
                reasons.append(f"PnL < ${min_pnl:,.0f}")
            if float(row["pnl_to_ui_vol_pct"]) < min_pv:
                reasons.append(f"P/V {row['pnl_to_ui_vol_pct']:.2f}% < {min_pv:.1f}%")

        if reasons:
            pre_fail_reasons[w] = reasons
        else:
            pre_pass_wallets.append(w)

    to_fetch = [w for w in pre_pass_wallets if w not in wallet_cache]
    done_cnt = len(pre_pass_wallets) - len(to_fetch)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=10) as executor:
            fut_to_w = {
                executor.submit(_probe_and_cache_positions, w, max_mkts, max_dead_open): w
                for w in to_fetch
            }
            for fut in as_completed(fut_to_w):
                w = fut_to_w[fut]
                try:
                    wallet_cache[w] = fut.result()
                except Exception as exc:
                    wallet_cache[w] = {
                        "settled_markets": 0,
                        "settled_capped": False,
                        "live_open": 0,
                        "dead_zero_open": 0,
                        "open_capped": False,
                        "recent_30d_mkts": 0,
                        "days_since_last": None,
                        "recent_market": f"Lỗi: {exc}",
                        "raw_positions": [],
                        "seed_acts": [],
                    }
                done_cnt += 1
                if progress_cb:
                    progress_cb(done_cnt, len(pre_pass_wallets))

    status_col, passed_flags, settled_col, open_summary_col = [], [], [], []
    recent_30d_col, fresh_col, recent_col, reason_col = [], [], [], []

    for _, row in work_df.iterrows():
        w = str(row["wallet"]).strip().lower()
        is_direct = bool(row.get("is_direct_lookup", False))
        if w in pre_fail_reasons:
            passed_flags.append(False)
            status_col.append("❌ LOẠI (Bề mặt)")
            settled_col.append("—")
            open_summary_col.append("—")
            recent_30d_col.append("—")
            fresh_col.append("—")
            recent_col.append("Bỏ qua gọi API (rớt P/V hoặc PnL)")
            reason_col.append(" | ".join(pre_fail_reasons[w]))
            continue

        info = wallet_cache.get(w, {})
        m_cnt = int(info.get("settled_markets") or 0)
        m_capped = bool(info.get("settled_capped", False))
        live_open = int(info.get("live_open") or 0)
        dead_open = int(info.get("dead_zero_open") or 0)
        open_capped = bool(info.get("open_capped", False))
        r30_cnt = int(info.get("recent_30d_mkts") or 0)
        d_last = info.get("days_since_last")

        reasons = []
        if not is_direct:
            if m_cnt < min_mkts:
                reasons.append(f"Ít mẫu đã đóng ({m_cnt} < {min_mkts} kèo)")
            elif m_capped or m_cnt > max_mkts:
                reasons.append(f"Quá trần kèo đã đóng (>{max_mkts})")

            if dead_open > max_dead_open:
                dead_disp = f"{dead_open}+" if open_capped else str(dead_open)
                reasons.append(f"Quá trần xác 0¢ ({dead_disp} > {max_dead_open})")

            if d_last is None:
                reasons.append("Không có dấu vết giao dịch gần đây")
            elif d_last > max_inactive_days:
                reasons.append(f"Ngừng vào lệnh ({d_last:.0f}d trước > {max_inactive_days:.0f}d)")

        settled_str = f">{max_mkts}" if m_capped else str(m_cnt)
        dead_lbl = f"{dead_open}+" if open_capped else str(dead_open)
        open_str = f"{live_open} sống / {dead_lbl} xác 0¢"
        fresh_str = f"{d_last:.1f}d trước" if d_last is not None else "Không rõ"

        settled_col.append(settled_str)
        open_summary_col.append(open_str)
        recent_30d_col.append(f"{r30_cnt} kèo" if r30_cnt > 0 else "0 kèo")
        fresh_col.append(fresh_str)
        recent_col.append(str(info.get("recent_market") or "N/A"))

        if not reasons:
            passed_flags.append(True)
            status_col.append("✅ ĐẠT BƯỚC 2")
            reason_col.append("Soi trực tiếp" if is_direct else "Đạt P/V, đủ mẫu kèo & đang chạy đều")
        else:
            passed_flags.append(False)
            status_col.append("❌ LOẠI (Bước 2)")
            reason_col.append(" | ".join(reasons))

    work_df["step2_passed"] = passed_flags
    work_df["step2_status"] = status_col
    work_df["settled_markets"] = settled_col
    work_df["open_summary"] = open_summary_col
    work_df["recent_30d"] = recent_30d_col
    work_df["last_active"] = fresh_col
    work_df["recent_market"] = recent_col
    work_df["step2_reason"] = reason_col

    df_passed = work_df[work_df["step2_passed"]].reset_index(drop=True)
    elapsed = round(time.perf_counter() - t0, 2)

    return df_passed, work_df, {
        "elapsed_s": elapsed,
        "input_count": len(work_df),
        "pre_pass_count": len(pre_pass_wallets),
        "passed_count": len(df_passed),
    }


# ==============================================================================
# BƯỚC 3: KIỂM TOÁN 4 LƯỚI LỌC CHỐNG NHIỄU (HOLD >= 80% + BAND 20-80% + SIZE/Z)
# ==============================================================================

def _build_ledger_from_raw(raw_pos: list[dict[str, Any]], raw_act: list[dict[str, Any]]) -> pd.DataFrame:
    if not raw_pos:
        return pd.DataFrame()

    pos_df = pd.DataFrame(raw_pos)

    def get_col(df: pd.DataFrame, names: list[str], default=None) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df), index=df.index)

    pos_df["condition_id"] = get_col(pos_df, ["condition_id", "conditionId"], "").astype(str).str.strip()
    pos_df["token_id"] = get_col(pos_df, ["token_id", "asset", "asset_id"], "").astype(str).str.strip()
    pos_df["title"] = get_col(pos_df, ["title", "question"], "N/A").astype(str)
    pos_df["outcome"] = get_col(pos_df, ["outcome"], "").astype(str)
    pos_df["raw_status"] = get_col(pos_df, ["status", "_queried_status"], "CLOSED").astype(str).str.upper()

    pos_df["avg_price"] = pd.to_numeric(
        get_col(pos_df, ["avg_price", "avgPrice", "average_price", "averagePrice", "buy_price", "entry_price"]),
        errors="coerce",
    ).fillna(0.0)
    pos_df["total_cost_usdc"] = pd.to_numeric(
        get_col(pos_df, ["total_cost_usdc", "totalCostUsdc", "cost_basis", "costBasis", "total_bought_usdc"]),
        errors="coerce",
    ).fillna(0.0)
    pos_df["entry_cost_usdc"] = pd.to_numeric(
        get_col(pos_df, ["entry_cost_usdc", "initialValue", "initial_value"]),
        errors="coerce",
    ).fillna(0.0)
    pos_df["total_size"] = pd.to_numeric(
        get_col(pos_df, ["total_size", "totalBought", "total_bought", "shares_bought"]),
        errors="coerce",
    ).fillna(0.0)
    pos_df["current_size"] = pd.to_numeric(get_col(pos_df, ["current_size", "size"]), errors="coerce").fillna(0.0)
    pos_df["cur_price"] = pd.to_numeric(get_col(pos_df, ["current_price", "curPrice", "price"]), errors="coerce").fillna(0.0)
    pos_df["current_value"] = pd.to_numeric(get_col(pos_df, ["current_value", "currentValue"]), errors="coerce").fillna(0.0)
    pos_df["realized_pnl"] = pd.to_numeric(get_col(pos_df, ["realized_pnl", "realizedPnl"]), errors="coerce").fillna(0.0)
    pos_df["unrealized_pnl"] = pd.to_numeric(get_col(pos_df, ["unrealized_pnl", "unrealizedPnl"]), errors="coerce").fillna(0.0)
    pos_df["api_total_pnl"] = pd.to_numeric(get_col(pos_df, ["total_pnl", "totalPnl"]), errors="coerce")

    pos_end_raw = get_col(pos_df, ["closed_at", "closedAt", "end_date", "endDate", "updated_at", "updatedAt"], None)
    pos_df["pos_end_ts"] = pos_end_raw.apply(_to_unix_ts)

    slug_s = get_col(pos_df, ["event_slug", "eventSlug", "slug"], "")
    pos_df["url"] = slug_s.apply(lambda s: f"https://polymarket.com/event/{s}" if pd.notna(s) and str(s).strip() else "")

    status_rank = {"OPEN": 1, "REDEEMABLE": 2, "REDEEMABLE_LOST": 2, "CLOSED": 3}
    pos_df["_srank"] = pos_df["raw_status"].map(status_rank).fillna(9)
    pos_df["_dedup_tok"] = np.where(pos_df["token_id"] != "", pos_df["token_id"], pos_df["condition_id"])
    pos_df = (
        pos_df.sort_values(["_srank", "total_cost_usdc", "entry_cost_usdc"], ascending=[True, False, False])
        .drop_duplicates(subset=["_dedup_tok", "outcome"], keep="first")
        .drop(columns=["_srank", "_dedup_tok"])
        .reset_index(drop=True)
    )

    is_dead_0c = (pos_df["raw_status"] == "OPEN") & (pos_df["cur_price"] <= 0.01) & (pos_df["current_value"] <= 0.5)
    pos_df["status"] = np.where(is_dead_0c, "SETTLED_LOSS_0C", pos_df["raw_status"])

    act_agg_by_tok: dict[str, dict[str, Any]] = {}
    merged_cids_from_act: set[str] = set()

    if raw_act:
        act_df = pd.DataFrame(raw_act)
        act_df["condition_id"] = get_col(act_df, ["condition_id", "conditionId", "market"], "").astype(str).str.strip()
        act_df["token_id"] = get_col(act_df, ["token_id", "asset", "asset_id"], "").astype(str).str.strip()
        act_df["type"] = get_col(act_df, ["type", "activity_type", "action"], "TRADE").astype(str).str.upper()
        act_df["side"] = get_col(act_df, ["side"], "").astype(str).str.upper()
        act_df["price"] = pd.to_numeric(get_col(act_df, ["price"]), errors="coerce").fillna(0.0)
        act_df["size"] = pd.to_numeric(get_col(act_df, ["size", "shares"]), errors="coerce").fillna(0.0)
        act_df["usdc"] = pd.to_numeric(get_col(act_df, ["usdc_size", "usdcSize", "amount", "value"]), errors="coerce").fillna(0.0)
        act_df["usdc"] = np.where(act_df["usdc"] > 0, act_df["usdc"], act_df["price"] * act_df["size"])
        ts_raw = get_col(act_df, ["timestamp", "created_at", "time"], None)
        act_df["ts"] = ts_raw.apply(_to_unix_ts)

        dedup_cols = [c for c in ["id", "transaction_hash", "transactionHash", "token_id", "ts", "side", "usdc"] if c in act_df.columns]
        if dedup_cols:
            act_df = act_df.drop_duplicates(subset=dedup_cols)

        merge_rows = act_df[act_df["type"].str.contains("MERGE|SPLIT", regex=True)]
        if not merge_rows.empty:
            merged_cids_from_act.update(c for c in merge_rows["condition_id"].unique() if c)

        for tid, grp in act_df.groupby("token_id"):
            if not tid:
                continue
            is_buy = (grp["side"] == "BUY") | ((grp["side"] == "") & grp["type"].str.contains("BUY"))
            is_sell = (grp["side"] == "SELL") | ((grp["side"] == "") & grp["type"].str.contains("SELL"))
            is_redeem = grp["type"].str.contains("REDEEM")
            is_merge = grp["type"].str.contains("MERGE")

            buys = grp[is_buy]
            sells = grp[is_sell]
            redeems = grp[is_redeem]
            merges = grp[is_merge]

            first_buy_ts = buys["ts"].min() if not buys.empty else grp["ts"].min()
            last_buy_ts = buys["ts"].max() if not buys.empty else grp["ts"].max()

            valid_buys_ts = buys[buys["ts"].notna()]
            if not valid_buys_ts.empty and valid_buys_ts["usdc"].sum() > 0:
                vwap_buy_ts = float((valid_buys_ts["ts"] * valid_buys_ts["usdc"]).sum() / valid_buys_ts["usdc"].sum())
            elif not valid_buys_ts.empty:
                vwap_buy_ts = float(valid_buys_ts["ts"].mean())
            else:
                vwap_buy_ts = first_buy_ts

            exit_candidates = []
            if not sells.empty and sells["ts"].notna().any():
                exit_candidates.append(float(sells["ts"].max()))
            if not redeems.empty and redeems["ts"].notna().any():
                exit_candidates.append(float(redeems["ts"].max()))
            if not merges.empty and merges["ts"].notna().any():
                exit_candidates.append(float(merges["ts"].max()))
            exit_ts = max(exit_candidates) if exit_candidates else np.nan

            accum_mins = (
                max(0.0, (last_buy_ts - first_buy_ts) / 60.0)
                if pd.notna(first_buy_ts) and pd.notna(last_buy_ts)
                else 0.0
            )
            act_hold_hours = (
                max(0.0, (exit_ts - vwap_buy_ts) / 3600.0)
                if pd.notna(vwap_buy_ts) and pd.notna(exit_ts) and (exit_ts > vwap_buy_ts + 60)
                else np.nan
            )

            valid_buy_prices = buys.loc[buys["price"] > 0, "price"]
            buy_price_spread_c = (
                float((valid_buy_prices.max() - valid_buy_prices.min()) * 100.0)
                if len(valid_buy_prices) >= 2
                else 0.0
            )
            act_w_price = (
                float((buys["price"] * buys["usdc"]).sum() / buys["usdc"].sum())
                if not buys.empty and buys["usdc"].sum() > 0
                else 0.0
            )

            act_agg_by_tok[tid] = {
                "has_act_match": True,
                "act_buy_usdc": float(buys["usdc"].sum()),
                "act_sell_usdc": float(sells["usdc"].sum()),
                "act_redeem_usdc": float(redeems["usdc"].sum()),
                "act_merge_usdc": float(merges["usdc"].sum()),
                "act_w_price": act_w_price,
                "buy_count": int(len(buys)),
                "sell_count": int(len(sells)),
                "merge_count": int(len(merges)),
                "accum_mins": accum_mins,
                "accum_hours": accum_mins / 60.0,
                "act_hold_hours": act_hold_hours,
                "buy_price_spread_c": buy_price_spread_c,
                "first_ts": first_buy_ts,
                "vwap_buy_ts": vwap_buy_ts,
                "exit_ts": exit_ts,
            }

    act_records = [act_agg_by_tok.get(tid, {}) for tid in pos_df["token_id"]]
    act_merged = pd.DataFrame(act_records, index=pos_df.index)

    for col, def_v in [
        ("has_act_match", False),
        ("act_buy_usdc", 0.0),
        ("act_sell_usdc", 0.0),
        ("act_redeem_usdc", 0.0),
        ("act_merge_usdc", 0.0),
        ("act_w_price", 0.0),
        ("buy_count", 0),
        ("sell_count", 0),
        ("merge_count", 0),
        ("accum_mins", 0.0),
        ("accum_hours", 0.0),
        ("act_hold_hours", np.nan),
        ("buy_price_spread_c", 0.0),
        ("first_ts", np.nan),
        ("vwap_buy_ts", np.nan),
        ("exit_ts", np.nan),
    ]:
        if col in act_merged.columns:
            if isinstance(def_v, bool):
                pos_df[col] = act_merged[col].where(act_merged[col].notna(), def_v).astype(bool)
            elif isinstance(def_v, int):
                pos_df[col] = pd.to_numeric(act_merged[col], errors="coerce").fillna(def_v).astype(int)
            else:
                pos_df[col] = pd.to_numeric(act_merged[col], errors="coerce").fillna(def_v).astype(float)
        else:
            pos_df[col] = def_v

    pos_df["avg_price"] = np.where(
        pos_df["avg_price"] > 0.001,
        pos_df["avg_price"],
        pos_df["act_w_price"],
    )

    entry_ref_ts = np.where(pos_df["vwap_buy_ts"].notna(), pos_df["vwap_buy_ts"], pos_df["first_ts"])
    fallback_hold_h = np.where(
        pd.notna(entry_ref_ts) & pos_df["pos_end_ts"].notna() & (pos_df["pos_end_ts"] > entry_ref_ts + 60),
        (pos_df["pos_end_ts"] - entry_ref_ts) / 3600.0,
        np.nan,
    )
    pos_df["hold_hours"] = np.where(pos_df["act_hold_hours"].notna(), pos_df["act_hold_hours"], fallback_hold_h)

    calc_pnl = pos_df["realized_pnl"] + pos_df["unrealized_pnl"]
    pos_df["total_pnl"] = np.where(
        pos_df["api_total_pnl"].notna() & (pos_df["api_total_pnl"].abs() > 1e-6),
        pos_df["api_total_pnl"],
        calc_pnl,
    )

    explicit_cost = np.where(
        pos_df["act_buy_usdc"] > 0,
        pos_df["act_buy_usdc"],
        np.where(
            pos_df["total_cost_usdc"] > 0,
            pos_df["total_cost_usdc"],
            np.where(pos_df["entry_cost_usdc"] > 0, pos_df["entry_cost_usdc"], 0.0),
        ),
    )

    raw_size_for_cost = np.where(
        pos_df["total_size"] > 0,
        pos_df["total_size"],
        np.where(pos_df["status"].isin(["OPEN", "SETTLED_LOSS_0C"]), pos_df["current_size"], 0.0),
    )
    size_cost = raw_size_for_cost * pos_df["avg_price"]

    known_prices = pos_df.loc[(pos_df["avg_price"] > 0.05) & (pos_df["avg_price"] < 0.95), "avg_price"]
    median_known_p = float(known_prices.median()) if len(known_prices) >= 5 else 0.55
    median_known_p = max(0.25, min(0.80, median_known_p))

    eff_price_for_infer = np.where(
        (pos_df["avg_price"] > 0.005) & (pos_df["avg_price"] < 0.995),
        pos_df["avg_price"],
        median_known_p,
    )

    inferred_cost = np.where(
        (pos_df["total_pnl"] < 0) & (pos_df["status"] != "OPEN"),
        pos_df["total_pnl"].abs(),
        np.where(
            (pos_df["total_pnl"] > 0) & (pos_df["status"] != "OPEN"),
            pos_df["total_pnl"] * eff_price_for_infer / (1.0 - eff_price_for_infer),
            0.0,
        ),
    )

    valid_size_cost = np.where(
        (pos_df["status"] != "OPEN")
        & (inferred_cost > 0)
        & (size_cost < 0.5 * inferred_cost),
        0.0,
        size_cost,
    )

    pos_df["true_cost"] = np.where(
        explicit_cost > 0,
        explicit_cost,
        np.where(valid_size_cost > 0, valid_size_cost, inferred_cost),
    )

    unclaimed_win = (
        (pos_df["raw_status"] == "REDEEMABLE")
        & (pos_df["total_pnl"].abs() < 1e-6)
        & (pos_df["current_value"] > pos_df["true_cost"])
        & (pos_df["true_cost"] > 0)
    )
    pos_df["total_pnl"] = np.where(unclaimed_win, pos_df["current_value"] - pos_df["true_cost"], pos_df["total_pnl"])

    open_est_val = np.where(
        pos_df["current_value"] > 0,
        pos_df["current_value"],
        pos_df["current_size"] * pos_df["cur_price"],
    )
    missing_open_pnl = (
        (pos_df["status"] == "OPEN")
        & (pos_df["total_pnl"].abs() < 1e-6)
        & (open_est_val > 0)
        & (pos_df["true_cost"] > 0)
    )
    pos_df["total_pnl"] = np.where(missing_open_pnl, open_est_val - pos_df["true_cost"], pos_df["total_pnl"])

    pos_df["true_cost"] = np.maximum(
        pos_df["true_cost"],
        np.where(pos_df["total_pnl"] < 0, pos_df["total_pnl"].abs(), 0.0),
    )

    zero_pnl_dead = (pos_df["status"] == "SETTLED_LOSS_0C") & (pos_df["total_pnl"].abs() < 1e-6) & (pos_df["true_cost"] > 0)
    pos_df["total_pnl"] = np.where(zero_pnl_dead, -pos_df["true_cost"], pos_df["total_pnl"])

    pos_df = pos_df[pos_df["true_cost"] >= 1.0].reset_index(drop=True)
    if pos_df.empty:
        return pos_df

    pos_df["avg_clip_usdc"] = np.where(
        pos_df["buy_count"] > 0,
        np.where(pos_df["act_buy_usdc"] > 0, pos_df["act_buy_usdc"], pos_df["true_cost"]) / pos_df["buy_count"],
        pos_df["true_cost"],
    )
    pos_df["roi_pct"] = (pos_df["total_pnl"] / pos_df["true_cost"] * 100.0).round(2)
    pos_df["real_win"] = (pos_df["status"] != "OPEN") & (pos_df["total_pnl"] > 0)

    cid_counts = pos_df[pos_df["condition_id"] != ""].groupby("condition_id")["outcome"].nunique()
    hedged_cids = set(cid_counts[cid_counts >= 2].index) | merged_cids_from_act
    pos_df["is_hedged"] = pos_df["condition_id"].isin(hedged_cids)

    has_act = pos_df["has_act_match"].to_numpy(dtype=bool)
    is_merged_row = ((pos_df["merge_count"] > 0) | (pos_df["act_merge_usdc"] > 0) | pos_df["condition_id"].isin(merged_cids_from_act)).to_numpy(dtype=bool)
    exit_conditions = [
        (pos_df["status"] == "OPEN").to_numpy(dtype=bool),
        is_merged_row,
        (pos_df["status"] == "SETTLED_LOSS_0C").to_numpy(dtype=bool),
        ~has_act,
        (pos_df["sell_count"] == 0).to_numpy(dtype=bool),
        ((pos_df["act_redeem_usdc"] <= 0) & (pos_df["act_sell_usdc"] > 0)).to_numpy(dtype=bool),
        ((pos_df["act_redeem_usdc"] > 0) & (pos_df["act_sell_usdc"] < 0.25 * np.maximum(pos_df["true_cost"], 1e-6))).to_numpy(dtype=bool),
    ]
    exit_choices = [
        "OPEN_LIVE",
        "MERGED_2_SIDES",
        "HOLD_TO_EXPIRY",
        "UNMATCHED_OLD",
        "HOLD_TO_EXPIRY",
        "SOLD_EARLY",
        "HOLD_TO_EXPIRY",
    ]
    pos_df["exit_style"] = np.select(exit_conditions, exit_choices, default="PARTIAL_SELL")

    dt_first = pd.to_datetime(pos_df["first_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    last_ref_ts = pos_df["exit_ts"].fillna(pos_df["pos_end_ts"]).fillna(pos_df["first_ts"])
    dt_last = pd.to_datetime(last_ref_ts, unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)

    pos_df["entry_time_vn"] = dt_first.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    pos_df["last_time_vn"] = dt_last.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    pos_df["month"] = dt_first.dt.strftime("%Y-%m").fillna("Unknown")

    pos_df["avg_price"] = pos_df["avg_price"].round(4)
    pos_df["true_cost"] = pos_df["true_cost"].round(2)
    pos_df["total_pnl"] = pos_df["total_pnl"].round(2)
    pos_df["avg_clip_usdc"] = pos_df["avg_clip_usdc"].round(2)
    pos_df["accum_mins"] = pos_df["accum_mins"].round(1)
    pos_df["hold_hours"] = pos_df["hold_hours"].round(2)
    return pos_df


def _evaluate_ledger_pillars(pos_df: pd.DataFrame, cfg: dict[str, Any], stage3a_only: bool = False) -> dict[str, Any]:
    """Đánh giá ví qua 4 Lưới Lọc Chống Nhiễu (HOLD >= 80%, Dải 20-80%, Size >= $300 & Z >= 1.65)."""
    empty_res = {
        "s3_verdict": "❌ KHÔNG DỮ LIỆU",
        "s3_passed": False,
        "passed_gates_cnt": 0,
        "gate_badge": "0/4 Lưới",
        "audit_pos_cnt": 0,
        "true_cost_usdc": 0.0,
        "ledger_pnl_usdc": 0.0,
        "open_pnl_usdc": 0.0,
        "true_roi_pct": 0.0,
        "count_wr_pct": 0.0,
        "cap_wr_pct": 0.0,
        "w_entry_pct": 0.0,
        "cap_edge_pct": 0.0,
        "z_score": 0.0,
        "med_pos_usdc": 0.0,
        "med_hold_h": 0.0,
        "live_cap_pct": 0.0,
        "med_clip_usdc": 0.0,
        "zero_sell_pct": 0.0,
        "hold_pnl_share_pct": 0.0,
        "act_match_info": "0/0",
        "core_band_pct": 0.0,
        "high_fav_pct": 0.0,
        "pnl_ex_top5": 0.0,
        "hedged_cap_pct": 0.0,
        "s3_flags": "Không có dữ liệu vị thế hợp lệ",
    }
    if pos_df.empty:
        return empty_res

    settled_df = pos_df[pos_df["status"] != "OPEN"].copy()
    if settled_df.empty:
        empty_res["s3_verdict"] = "⏳ CHƯA CÓ KÈO ĐÓNG"
        empty_res["act_match_info"] = f"0/{len(pos_df)} (OPEN)"
        empty_res["s3_flags"] = "Ví chỉ có vị thế đang OPEN, chưa có lịch sử đã ngã ngũ"
        return empty_res

    n_settled = len(settled_df)
    total_cost = float(settled_df["true_cost"].sum())
    net_pnl = float(settled_df["total_pnl"].sum())
    true_roi = (net_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

    open_df = pos_df[pos_df["status"] == "OPEN"]
    open_pnl = float(open_df["total_pnl"].sum()) if not open_df.empty else 0.0
    all_in_pnl = net_pnl + open_pnl

    known_p = settled_df.loc[(settled_df["avg_price"] > 0.005) & (settled_df["avg_price"] < 0.995), "avg_price"]
    fallback_p = float(known_p.median()) if not known_p.empty else 0.50
    eff_p = np.where(
        (settled_df["avg_price"] > 0.005) & (settled_df["avg_price"] < 0.995),
        settled_df["avg_price"],
        fallback_p,
    )

    payout_shares = np.where(
        settled_df["real_win"],
        np.maximum(settled_df["true_cost"] + settled_df["total_pnl"], settled_df["true_cost"] / eff_p),
        settled_df["true_cost"] / eff_p,
    )
    tot_shares = float(payout_shares.sum())
    win_shares = float(payout_shares[settled_df["real_win"]].sum())

    count_wr = float(settled_df["real_win"].mean() * 100.0)
    cap_wr = (win_shares / tot_shares * 100.0) if tot_shares > 0 else 0.0
    w_entry = (total_cost / tot_shares * 100.0) if tot_shares > 0 else 50.0
    cap_edge = cap_wr - w_entry

    # Kiểm định thống kê Z-score trọng số vốn (gom theo condition_id tránh nhân đôi mẫu)
    settled_df["_share_std"] = payout_shares * np.sqrt(eff_p * (1.0 - eff_p))
    mkt_stat_grp = settled_df.groupby(
        np.where(settled_df["condition_id"] != "", settled_df["condition_id"], settled_df["token_id"])
    ).agg(mkt_pnl=("total_pnl", "sum"), mkt_std=("_share_std", "sum"))
    denom_std = float(np.sqrt((mkt_stat_grp["mkt_std"] ** 2).sum()))
    z_score = (net_pnl / denom_std) if denom_std > 1e-6 else 0.0

    priced_df = settled_df[settled_df["avg_price"] > 0.005]
    priced_cost = float(priced_df["true_cost"].sum())

    matched_settled = settled_df[settled_df["exit_style"].isin(["HOLD_TO_EXPIRY", "SOLD_EARLY", "PARTIAL_SELL", "MERGED_2_SIDES"])]
    act_matched_only = settled_df[settled_df["has_act_match"]]

    hold_df = matched_settled[matched_settled["exit_style"] == "HOLD_TO_EXPIRY"]
    early_df = matched_settled[matched_settled["exit_style"].isin(["SOLD_EARLY", "PARTIAL_SELL", "MERGED_2_SIDES"])]

    hold_pnl = float(hold_df["total_pnl"].sum())
    early_pnl = float(early_df["total_pnl"].sum())
    matched_net_pnl = float(matched_settled["total_pnl"].sum())

    if len(matched_settled) == 0:
        hold_pnl_share = 0.0
    elif matched_net_pnl > 0:
        hold_pnl_share = max(-100.0, min(100.0, hold_pnl / matched_net_pnl * 100.0))
    else:
        pos_hold = float(hold_df.loc[hold_df["total_pnl"] > 0, "total_pnl"].sum())
        pos_early = float(early_df.loc[early_df["total_pnl"] > 0, "total_pnl"].sum())
        tot_pos = pos_hold + pos_early
        hold_pnl_share = (pos_hold / tot_pos * 100.0) if tot_pos > 0 else 0.0

    # --- Lưới 1 & 2: Timing Pre-match, Quy mô kèo & Diamond Hands ---
    buy_matched = pos_df[pos_df["buy_count"] > 0]
    zero_sell_pct = float((act_matched_only["sell_count"] == 0).mean() * 100.0) if not act_matched_only.empty else 0.0
    med_clip_usdc = float(buy_matched["avg_clip_usdc"].median()) if not buy_matched.empty else 0.0
    med_pos_usdc = float(settled_df["true_cost"].median())

    live_cutoff_h = float(cfg["live_cutoff_hours"])
    valid_holds = pos_df[pos_df["hold_hours"].notna()].copy()
    if not valid_holds.empty:
        med_hold_h = float(valid_holds["hold_hours"].median())
        vh_cost = float(valid_holds["true_cost"].sum())
        live_cap_pct = (
            float(valid_holds.loc[valid_holds["hold_hours"] < live_cutoff_h, "true_cost"].sum() / vh_cost * 100.0)
            if vh_cost > 0
            else 0.0
        )
    else:
        med_hold_h = 0.0
        live_cap_pct = 0.0

    # --- Lưới 4: Dải xác suất 20-80% & Rủi ro ---
    if priced_cost > 0:
        core_band_pct = float(
            priced_df[(priced_df["avg_price"] >= 0.20) & (priced_df["avg_price"] <= 0.80)]["true_cost"].sum()
            / priced_cost
            * 100.0
        )
        high_fav_pct = float(priced_df[priced_df["avg_price"] >= 0.85]["true_cost"].sum() / priced_cost * 100.0)
    else:
        core_band_pct = 100.0
        high_fav_pct = 0.0

    mkt_pnls = mkt_stat_grp["mkt_pnl"]
    top5_sum = float(mkt_pnls[mkt_pnls > 0].sort_values(ascending=False).head(5).sum())
    pnl_ex_top5 = net_pnl - top5_sum
    hedged_cap_pct = (
        (float(settled_df.loc[settled_df["is_hedged"], "true_cost"].sum()) / total_cost * 100.0)
        if total_cost > 0
        else 0.0
    )

    # ==========================================================================
    # ĐÁNH GIÁ 4 LƯỚI LỌC CHỐNG NHIỄU
    # ==========================================================================
    failed_gates: list[str] = []
    passed_cnt = 0

    # Lưới 1: Chống Trade Live, Mua sát giờ & Chặn ví lệnh siêu nhỏ
    g1_reasons = []
    if len(act_matched_only) < int(cfg["min_act_matched"]):
        g1_reasons.append(f"ít mẫu khớp Act ({len(act_matched_only)}<{cfg['min_act_matched']})")
    if med_pos_usdc < float(cfg["min_med_pos_usdc"]):
        g1_reasons.append(f"size kèo TB bé (${med_pos_usdc:,.0f}<${cfg['min_med_pos_usdc']:,.0f})")
    if len(valid_holds) >= 5 and med_hold_h < float(cfg["min_hold_hours"]):
        g1_reasons.append(f"vào sát giờ/Live (Hold TB {med_hold_h:.1f}h<{cfg['min_hold_hours']:.1f}h)")
    if len(valid_holds) >= 5 and live_cap_pct > float(cfg["max_live_cap_pct"]):
        g1_reasons.append(f"vốn vào <{live_cutoff_h:.1f}h cao ({live_cap_pct:.1f}%>{cfg['max_live_cap_pct']:.1f}%)")
    if not g1_reasons and not stage3a_only:
        passed_cnt += 1
    else:
        failed_gates.append("L1-Timing/Size: " + (", ".join(g1_reasons) if g1_reasons else "chưa kéo Act"))

    # Lưới 2: Kỷ luật giữ đáo hạn (Diamond Hands >= 80%)
    g2_reasons = []
    if zero_sell_pct < float(cfg["min_zero_sell_pct"]):
        g2_reasons.append(f"hay bán non (ZeroSell={zero_sell_pct:.0f}%<{cfg['min_zero_sell_pct']:.0f}%)")
    if hold_pnl_share < float(cfg["min_hold_redeem"]):
        g2_reasons.append(f"lãi bán non/merge (Hold={hold_pnl_share:.0f}%<{cfg['min_hold_redeem']:.0f}%, NonHold=${early_pnl:+,.0f})")
    if not g2_reasons and not stage3a_only:
        passed_cnt += 1
    else:
        failed_gates.append("L2-Đáo hạn: " + (", ".join(g2_reasons) if g2_reasons else "chưa kéo Act"))

    # Lưới 3: Lợi thế dự báo thực & Kiểm định thống kê (Z >= 1.65)
    g3_reasons = []
    if true_roi < float(cfg["min_true_roi"]):
        g3_reasons.append(f"ROI {true_roi:+.1f}%<{cfg['min_true_roi']:.1f}%")
    if cap_edge < float(cfg["min_cap_edge"]):
        g3_reasons.append(f"TrueEdge {cap_edge:+.1f}%<{cfg['min_cap_edge']:.1f}%")
    if z_score < float(cfg["min_z_score"]):
        g3_reasons.append(f"Z={z_score:.2f}<{cfg['min_z_score']:.2f}")
    if not g3_reasons:
        passed_cnt += 1
    else:
        failed_gates.append("L3-Edge: " + ", ".join(g3_reasons))

    # Lưới 4: Khóa dải xác suất 20-80% & Quản trị rủi ro
    g4_reasons = []
    if w_entry > float(cfg["max_w_entry"]):
        g4_reasons.append(f"giá mua TB cao ({w_entry:.1f}%>{cfg['max_w_entry']:.0f}%)")
    if core_band_pct < float(cfg["min_core_band"]):
        g4_reasons.append(f"dải 20-80% thấp ({core_band_pct:.0f}%<{cfg['min_core_band']:.0f}%)")
    if high_fav_pct > float(cfg["max_high_fav"]):
        g4_reasons.append(f"bẫy cửa trên >=85¢ ({high_fav_pct:.1f}%>{cfg['max_high_fav']:.0f}%)")
    if hedged_cap_pct > float(
