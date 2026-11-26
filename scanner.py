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
    page_title="Polymarket Pro Wallet Finder — 4-Pillar Pipeline (v5.3.1)",
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
    "Esports (CS2, Dota 2, LoL, Valorant...)": "ESPORTS",
    "Sports (NBA, Tennis, Bóng đá, UFC...)": "SPORTS",
    "Finance (Chỉ số, Cổ phiếu, Vàng, Hàng hóa)": "FINANCE",
    "Economics (Lãi suất Fed, CPI, GDP, Vĩ mô)": "ECONOMICS",
    "Weather (Thời tiết & Nhiệt độ)": "WEATHER",
    "Politics (Chính trị & Bầu cử)": "POLITICS",
    "Crypto (Tiền mã hóa)": "CRYPTO",
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
                "User-Agent": "Polymarket-Wallet-Finder/5.3.1-4Pillars",
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
    """Trả về (json_payload, http_status_code) để chẩn đoán chính xác kể cả khi data rỗng."""
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
# BƯỚC 1: THU THẬP TOP N VÍ (CHIA LÔ AN TOÀN 20/TRANG + LOG CHẨN ĐOÁN RÕ)
# ==============================================================================

def step1_fetch_leaderboard(
    category: str,
    time_period: str,
    order_by: str = "PNL",
    top_n: int = 40,
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
    allow_live_open_override = bool(cfg["allow_live_open_override"])

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

            is_fresh = False
            if d_last is not None and d_last <= max_inactive_days:
                is_fresh = True
            elif allow_live_open_override and live_open > 0:
                is_fresh = True

            if not is_fresh:
                if d_last is None:
                    reasons.append("Không có dấu vết hoạt động gần đây")
                else:
                    reasons.append(f"Ngừng chạy ({d_last:.0f}d trước > {max_inactive_days:.0f}d)")

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
            reason_col.append("Soi trực tiếp" if is_direct else "Đạt P/V, đủ mẫu kèo & đang chạy")
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
# BƯỚC 3: KIỂM TOÁN 4 LƯỚI LỌC TINH GỌN (SHARE-WEIGHTED TRUE EDGE)
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
    if raw_act:
        act_df = pd.DataFrame(raw_act)
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

        for tid, grp in act_df.groupby("token_id"):
            if not tid:
                continue
            is_buy = (grp["side"] == "BUY") | ((grp["side"] == "") & grp["type"].str.contains("BUY"))
            is_sell = (grp["side"] == "SELL") | ((grp["side"] == "") & grp["type"].str.contains("SELL"))
            is_redeem = grp["type"].str.contains("REDEEM")

            buys = grp[is_buy]
            sells = grp[is_sell]
            redeems = grp[is_redeem]

            first_buy_ts = buys["ts"].min() if not buys.empty else grp["ts"].min()
            last_buy_ts = buys["ts"].max() if not buys.empty else grp["ts"].max()

            # Thời điểm vào lệnh tính theo trọng số vốn (chống lệnh mồi sớm rồi nhồi Live)
            valid_buys_ts = buys[buys["ts"].notna()]
            if not valid_buys_ts.empty and valid_buys_ts["usdc"].sum() > 0:
                vwap_buy_ts = float((valid_buys_ts["ts"] * valid_buys_ts["usdc"]).sum() / valid_buys_ts["usdc"].sum())
            elif not valid_buys_ts.empty:
                vwap_buy_ts = float(valid_buys_ts["ts"].mean())
            else:
                vwap_buy_ts = first_buy_ts

            # Chỉ tính thời gian thoát từ Activity khi thực sự có SELL hoặc REDEEM
            exit_candidates = []
            if not sells.empty and sells["ts"].notna().any():
                exit_candidates.append(float(sells["ts"].max()))
            if not redeems.empty and redeems["ts"].notna().any():
                exit_candidates.append(float(redeems["ts"].max()))
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
                "act_w_price": act_w_price,
                "buy_count": int(len(buys)),
                "sell_count": int(len(sells)),
                "accum_mins": accum_mins,
                "accum_hours": accum_mins / 60.0,
                "act_hold_hours": act_hold_hours,
                "buy_price_spread_c": buy_price_spread_c,
                "first_ts": first_buy_ts,
                "vwap_buy_ts": vwap_buy_ts,
            }

    act_records = [act_agg_by_tok.get(tid, {}) for tid in pos_df["token_id"]]
    act_merged = pd.DataFrame(act_records, index=pos_df.index)

    for col, def_v in [
        ("has_act_match", False),
        ("act_buy_usdc", 0.0),
        ("act_sell_usdc", 0.0),
        ("act_redeem_usdc", 0.0),
        ("act_w_price", 0.0),
        ("buy_count", 0),
        ("sell_count", 0),
        ("accum_mins", 0.0),
        ("accum_hours", 0.0),
        ("act_hold_hours", np.nan),
        ("buy_price_spread_c", 0.0),
        ("first_ts", np.nan),
        ("vwap_buy_ts", np.nan),
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
    pos_df["roi_pct"] = pos_df["total_pnl"] / pos_df["true_cost"] * 100.0
    pos_df["real_win"] = (pos_df["status"] != "OPEN") & (pos_df["total_pnl"] > 0)

    cid_counts = pos_df[pos_df["condition_id"] != ""].groupby("condition_id")["outcome"].nunique()
    hedged_cids = set(cid_counts[cid_counts >= 2].index)
    pos_df["is_hedged"] = pos_df["condition_id"].isin(hedged_cids)

    has_act = pos_df["has_act_match"].to_numpy(dtype=bool)
    exit_conditions = [
        (pos_df["status"] == "OPEN").to_numpy(dtype=bool),
        (pos_df["status"] == "SETTLED_LOSS_0C").to_numpy(dtype=bool),
        ~has_act,
        (pos_df["sell_count"] == 0).to_numpy(dtype=bool),
        ((pos_df["act_redeem_usdc"] <= 0) & (pos_df["act_sell_usdc"] > 0)).to_numpy(dtype=bool),
        ((pos_df["act_redeem_usdc"] > 0) & (pos_df["act_sell_usdc"] < 0.25 * np.maximum(pos_df["true_cost"], 1e-6))).to_numpy(dtype=bool),
    ]
    exit_choices = [
        "OPEN_LIVE",
        "HOLD_TO_EXPIRY",
        "UNMATCHED_OLD",
        "HOLD_TO_EXPIRY",
        "SOLD_EARLY",
        "HOLD_TO_EXPIRY",
    ]
    pos_df["exit_style"] = np.select(exit_conditions, exit_choices, default="PARTIAL_SELL")
    pos_df["entry_time_vn"] = pd.to_datetime(pos_df["first_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    return pos_df


def _evaluate_ledger_pillars(pos_df: pd.DataFrame, cfg: dict[str, Any], stage3a_only: bool = False) -> dict[str, Any]:
    """Đánh giá ví qua 4 Lưới Lọc Tinh Gọn với Share-Weighted True Edge."""
    if pos_df.empty:
        return {
            "s3_verdict": "❌ KHÔNG DỮ LIỆU",
            "s3_passed": False,
            "passed_gates_cnt": 0,
            "gate_badge": "0/4 Lưới",
            "audit_pos_cnt": 0,
            "true_cost_usdc": 0.0,
            "ledger_pnl_usdc": 0.0,
            "true_roi_pct": 0.0,
            "count_wr_pct": 0.0,
            "cap_wr_pct": 0.0,
            "w_entry_pct": 0.0,
            "cap_edge_pct": 0.0,
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

    settled_df = pos_df[pos_df["status"] != "OPEN"].copy()
    if settled_df.empty:
        return {
            "s3_verdict": "⏳ CHƯA CÓ KÈO ĐÓNG",
            "s3_passed": False,
            "passed_gates_cnt": 0,
            "gate_badge": "0/4 Lưới",
            "audit_pos_cnt": 0,
            "true_cost_usdc": 0.0,
            "ledger_pnl_usdc": 0.0,
            "true_roi_pct": 0.0,
            "count_wr_pct": 0.0,
            "cap_wr_pct": 0.0,
            "w_entry_pct": 0.0,
            "cap_edge_pct": 0.0,
            "med_hold_h": 0.0,
            "live_cap_pct": 0.0,
            "med_clip_usdc": 0.0,
            "zero_sell_pct": 0.0,
            "hold_pnl_share_pct": 0.0,
            "act_match_info": f"0/{len(pos_df)} (OPEN)",
            "core_band_pct": 0.0,
            "high_fav_pct": 0.0,
            "pnl_ex_top5": 0.0,
            "hedged_cap_pct": 0.0,
            "s3_flags": "Ví chỉ có vị thế đang OPEN, chưa có lịch sử đã ngã ngũ",
        }

    n_settled = len(settled_df)
    total_cost = float(settled_df["true_cost"].sum())
    net_pnl = float(settled_df["total_pnl"].sum())
    true_roi = (net_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

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

    priced_df = settled_df[settled_df["avg_price"] > 0.005]
    priced_cost = float(priced_df["true_cost"].sum())

    matched_settled = settled_df[settled_df["exit_style"].isin(["HOLD_TO_EXPIRY", "SOLD_EARLY", "PARTIAL_SELL"])]
    act_matched_only = settled_df[settled_df["has_act_match"]]

    hold_df = matched_settled[matched_settled["exit_style"] == "HOLD_TO_EXPIRY"]
    early_df = matched_settled[matched_settled["exit_style"] == "SOLD_EARLY"]

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

    # --- Lưới 1 & 2: Timing Pre-match & Diamond Hands ---
    act_matched_all = pos_df[pos_df["has_act_match"]]
    buy_matched = pos_df[pos_df["buy_count"] > 0]
    zero_sell_pct = float((act_matched_all["sell_count"] == 0).mean() * 100.0) if not act_matched_all.empty else 0.0
    med_clip_usdc = float(buy_matched["avg_clip_usdc"].median()) if not buy_matched.empty else 0.0

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

    # --- Lưới 4: Dải xác suất & Rủi ro ---
    if priced_cost > 0:
        core_band_pct = float(
            priced_df[(priced_df["avg_price"] >= 0.15) & (priced_df["avg_price"] <= 0.90)]["true_cost"].sum()
            / priced_cost
            * 100.0
        )
        high_fav_pct = float(priced_df[priced_df["avg_price"] >= 0.95]["true_cost"].sum() / priced_cost * 100.0)
    else:
        core_band_pct = 100.0
        high_fav_pct = 0.0

    mkt_pnls = settled_df.groupby(
        np.where(settled_df["condition_id"] != "", settled_df["condition_id"], settled_df["token_id"])
    )["total_pnl"].sum()
    top5_sum = float(mkt_pnls[mkt_pnls > 0].sort_values(ascending=False).head(5).sum())
    pnl_ex_top5 = net_pnl - top5_sum
    hedged_cap_pct = (
        (float(settled_df.loc[settled_df["is_hedged"], "true_cost"].sum()) / total_cost * 100.0)
        if total_cost > 0
        else 0.0
    )

    # ==========================================================================
    # ĐÁNH GIÁ 4 LƯỚI LỌC TINH GỌN
    # ==========================================================================
    failed_gates: list[str] = []
    passed_cnt = 0

    # Lưới 1: Chống Trade Live & Mua sát giờ thi đấu
    g1_reasons = []
    if len(act_matched_only) < int(cfg["min_act_matched"]):
        g1_reasons.append(f"ít mẫu khớp Act ({len(act_matched_only)}<{cfg['min_act_matched']})")
    if len(valid_holds) >= 5 and med_hold_h < float(cfg["min_hold_hours"]):
        g1_reasons.append(f"vào sát giờ/Live (Hold TB {med_hold_h:.1f}h<{cfg['min_hold_hours']:.1f}h)")
    if len(valid_holds) >= 5 and live_cap_pct > float(cfg["max_live_cap_pct"]):
        g1_reasons.append(f"vốn vào <{live_cutoff_h:.1f}h cao ({live_cap_pct:.1f}%>{cfg['max_live_cap_pct']:.1f}%)")
    if not g1_reasons and not stage3a_only:
        passed_cnt += 1
    else:
        failed_gates.append("L1-Timing: " + (", ".join(g1_reasons) if g1_reasons else "chưa kéo Act"))

    # Lưới 2: Kỷ luật giữ đáo hạn (Diamond Hands)
    g2_reasons = []
    if zero_sell_pct < float(cfg["min_zero_sell_pct"]):
        g2_reasons.append(f"hay bán non (ZeroSell={zero_sell_pct:.0f}%<{cfg['min_zero_sell_pct']:.0f}%)")
    if hold_pnl_share < float(cfg["min_hold_redeem"]):
        g2_reasons.append(f"lãi bán non (Hold={hold_pnl_share:.0f}%<{cfg['min_hold_redeem']:.0f}%, Early=${early_pnl:+,.0f})")
    if not g2_reasons and not stage3a_only:
        passed_cnt += 1
    else:
        failed_gates.append("L2-Đáo hạn: " + (", ".join(g2_reasons) if g2_reasons else "chưa kéo Act"))

    # Lưới 3: Lợi thế dự báo thực (True ROI & Share-Weighted Edge)
    g3_reasons = []
    if true_roi < float(cfg["min_true_roi"]):
        g3_reasons.append(f"ROI {true_roi:+.1f}%<{cfg['min_true_roi']:.1f}%")
    if cap_edge < float(cfg["min_cap_edge"]):
        g3_reasons.append(f"TrueEdge {cap_edge:+.1f}%<{cfg['min_cap_edge']:.1f}%")
    if not g3_reasons:
        passed_cnt += 1
    else:
        failed_gates.append("L3-Edge: " + ", ".join(g3_reasons))

    # Lưới 4: Dải xác suất & Quản trị rủi ro
    g4_reasons = []
    if core_band_pct < float(cfg["min_core_band"]):
        g4_reasons.append(f"dải 15-90% thấp ({core_band_pct:.0f}%<{cfg['min_core_band']:.0f}%)")
    if high_fav_pct > float(cfg["max_high_fav"]):
        g4_reasons.append(f"bẫy Fav >=95¢ ({high_fav_pct:.1f}%>{cfg['max_high_fav']:.0f}%)")
    if hedged_cap_pct > float(cfg["max_hedged_pct"]):
        g4_reasons.append(f"vốn hedge 2 đầu ({hedged_cap_pct:.1f}%>{cfg['max_hedged_pct']:.0f}%)")
    if bool(cfg["require_ex_top5"]) and pnl_ex_top5 <= 0:
        g4_reasons.append(f"ex-Top5=${pnl_ex_top5:+,.0f}<=0")
    if not g4_reasons:
        passed_cnt += 1
    else:
        failed_gates.append("L4-Rủi ro: " + ", ".join(g4_reasons))

    if stage3a_only:
        verdict = "❌ LOẠI (3A Âm ROI)"
        passed = False
    elif passed_cnt == 4:
        verdict = "🏆 CHUẨN PRO (4/4)"
        passed = True
    elif passed_cnt == 3:
        verdict = "🟡 SÁT CHUẨN (3/4)"
        passed = bool(cfg["allow_3_of_4_pro"])
    else:
        verdict = f"❌ LOẠI ({passed_cnt}/4 Lưới)"
        passed = False

    return {
        "s3_verdict": verdict,
        "s3_passed": passed,
        "passed_gates_cnt": passed_cnt,
        "gate_badge": "Dừng 3A" if stage3a_only else f"{passed_cnt}/4 Lưới",
        "audit_pos_cnt": n_settled,
        "true_cost_usdc": round(total_cost, 2),
        "ledger_pnl_usdc": round(net_pnl, 2),
        "true_roi_pct": round(true_roi, 2),
        "count_wr_pct": round(count_wr, 1),
        "cap_wr_pct": round(cap_wr, 1),
        "w_entry_pct": round(w_entry, 1),
        "cap_edge_pct": round(cap_edge, 2),
        "med_hold_h": round(med_hold_h, 1),
        "live_cap_pct": round(live_cap_pct, 1),
        "med_clip_usdc": round(med_clip_usdc, 0),
        "zero_sell_pct": round(zero_sell_pct, 0),
        "hold_pnl_share_pct": round(hold_pnl_share, 1),
        "act_match_info": "Dừng 3A" if stage3a_only else f"{len(act_matched_only)}/{n_settled}",
        "core_band_pct": round(core_band_pct, 1),
        "high_fav_pct": round(high_fav_pct, 1),
        "pnl_ex_top5": round(pnl_ex_top5, 2),
        "hedged_cap_pct": round(hedged_cap_pct, 1),
        "s3_flags": " | ".join(failed_gates) if failed_gates else "✅ Đạt trọn vẹn 4/4 Lưới chuẩn Copy-Trade",
    }


def step3_deep_ledger_audit(
    df_step2_passed: pd.DataFrame,
    cfg: dict[str, Any],
    wallet_cache: dict[str, dict[str, Any]],
    activity_cache: dict[str, list[dict[str, Any]]],
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    t0 = time.perf_counter()
    if df_step2_passed.empty:
        return (
            df_step2_passed.copy(),
            df_step2_passed.copy(),
            {},
            {"elapsed_s": 0.0, "input_count": 0, "stage3b_count": 0, "passed_count": 0, "watch_count": 0},
        )

    work_df = df_step2_passed.copy()
    wallets = [str(w).strip().lower() for w in work_df["wallet"].tolist()]
    direct_set = set(
        str(r["wallet"]).strip().lower()
        for _, r in work_df.iterrows()
        if bool(r.get("is_direct_lookup", False))
    )
    max_act_pages = int(cfg["max_act_pages"])

    audit_map: dict[str, dict[str, Any]] = {}
    details_map: dict[str, pd.DataFrame] = {}
    stage3b_candidates: list[str] = []

    for w in wallets:
        cached = wallet_cache.get(w, {})
        raw_pos = cached.get("raw_positions") or []
        seed_acts = cached.get("seed_acts") or []

        pre_df = _build_ledger_from_raw(raw_pos, seed_acts)
        pre_eval = _evaluate_ledger_pillars(pre_df, cfg, stage3a_only=True)

        fail_hard_3a = (w not in direct_set) and (
            pre_df.empty
            or pre_eval["true_roi_pct"] < -5.0
        )

        if fail_hard_3a:
            audit_map[w] = pre_eval
            if not pre_df.empty:
                details_map[w] = pre_df
        else:
            stage3b_candidates.append(w)
            if w not in activity_cache or not activity_cache[w]:
                all_matched = (not pre_df.empty) and bool(pre_df["has_act_match"].all())
                if (0 < len(seed_acts) < 100) or all_matched:
                    activity_cache[w] = seed_acts

    to_fetch_act = [
        w for w in stage3b_candidates
        if w not in activity_cache or not activity_cache[w]
    ]
    done_3b = len(stage3b_candidates) - len(to_fetch_act)

    if to_fetch_act:
        with ThreadPoolExecutor(max_workers=10) as executor:
            fut_to_w = {executor.submit(_fetch_activities_v2, w, max_act_pages): w for w in to_fetch_act}
            for fut in as_completed(fut_to_w):
                w = fut_to_w[fut]
                try:
                    fetched_acts = fut.result()
                    activity_cache[w] = fetched_acts if fetched_acts else (wallet_cache.get(w, {}).get("seed_acts") or [])
                except Exception:
                    activity_cache[w] = wallet_cache.get(w, {}).get("seed_acts") or []
                done_3b += 1
                if progress_cb:
                    progress_cb(done_3b, len(stage3b_candidates))

    for w in stage3b_candidates:
        raw_pos = wallet_cache.get(w, {}).get("raw_positions") or []
        full_acts = activity_cache.get(w) or wallet_cache.get(w, {}).get("seed_acts") or []
        full_df = _build_ledger_from_raw(raw_pos, full_acts)
        audit_map[w] = _evaluate_ledger_pillars(full_df, cfg, stage3a_only=False)
        if not full_df.empty:
            details_map[w] = full_df

    attach_cols = list(next(iter(audit_map.values())).keys())
    for c in attach_cols:
        work_df[c] = [audit_map[str(w).strip().lower()][c] for w in work_df["wallet"]]

    work_df = work_df.sort_values(
        by=["s3_passed", "passed_gates_cnt", "cap_edge_pct", "true_roi_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    df_final = work_df[work_df["s3_passed"]].reset_index(drop=True)
    watch_cnt = int((work_df["passed_gates_cnt"] == 3).sum())
    elapsed = round(time.perf_counter() - t0, 2)

    return (
        df_final,
        work_df,
        details_map,
        {
            "elapsed_s": elapsed,
            "input_count": len(work_df),
            "stage3b_count": len(stage3b_candidates),
            "passed_count": len(df_final),
            "watch_count": watch_cnt,
        },
    )


# ==============================================================================
# GIAO DIỆN STREAMLIT ĐIỀU KHIỂN FULL PIPELINE 4 LƯỚI LỌC TINH GỌN (v5.3.1)
# ==============================================================================

st.title("📡 Polymarket Pro Wallet Finder — 4-Pillar Pipeline (v5.3.1)")
st.caption(
    "**Bước 1:** Kéo Leaderboard / Soi trực tiếp `0x...` → "
    "**Bước 2:** Lọc bề mặt & cache sổ vị thế → "
    "**Bước 3:** Kiểm toán **4 Lưới Cốt Lõi** (Pre-match Timing • Diamond Hands • Share-Weighted Edge • Rủi Ro)."
)

s1, s2, s3 = st.columns(3)
s1.success("✅ **Bước 1: Thu thập Leaderboard**\n\nChia lô an toàn + Soi trực tiếp ví `0x...`")
s2.success("✅ **Bước 2: Lọc & Cache Positions**\n\nVá chuẩn vốn các kèo thắng lịch sử")
s3.success("✅ **Bước 3: 4 Lưới Lọc Tinh Gọn**\n\nChặn Live/Sát giờ • Diamond Hands • TrueEdge")

for key, default_val in [
    ("df_step1_raw", pd.DataFrame()),
    ("df_final_pros", pd.DataFrame()),
    ("df_step3_all", pd.DataFrame()),
    ("df_step2_all", pd.DataFrame()),
    ("step3_details_map", {}),
    ("wallet_cache", {}),
    ("activity_cache", {}),
    ("pipeline_meta", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

with st.sidebar:
    st.header("⚙️ Bước 2: Lưới Lọc Bề Mặt")
    min_pv_pct = st.slider("1. Sàn PnL / UI Volume (%)", 0.0, 15.0, 3.0, 0.5)
    min_net_pnl = st.number_input("2. Net PnL tối thiểu ($)", min_value=0.0, value=3000.0, step=1000.0)

    c_m1, c_m2 = st.columns(2)
    min_markets = c_m1.number_input("3. Kèo đóng tối thiểu", min_value=1, max_value=500, value=50, step=10)
    max_markets = c_m2.number_input("4. Kèo đóng tối đa", min_value=50, max_value=10000, value=4000, step=250)

    max_dead_open = st.number_input("5. Trần xác vị thế 0¢ tối đa", min_value=0, max_value=5000, value=1500, step=100)
    max_inactive_days = st.slider("6. Ngừng chạy tối đa (ngày)", 1, 90, 30, 1)
    allow_live_open_override = st.checkbox("Giữ ví nếu đang có kèo chưa ngã ngũ (> $1)", value=True)

    st.divider()
    st.header("🔬 Bước 3: 4 Lưới Lọc Sổ Cái")
    allow_3_of_4_pro = st.checkbox("Chấp nhận ví đạt 3/4 Lưới vào danh sách Pro", value=False)

    with st.expander("⏱️ Lưới 1 & 2: Chống Live & Diamond Hands", expanded=True):
        min_hold_hours = st.slider(
            "Hold trung vị tối thiểu (giờ)",
            1.0,
            24.0,
            3.0,
            0.5,
            help="Trận đấu thường kéo dài ~1.5–2h. Đặt >= 3.0h để loại bỏ nhóm đánh Live hoặc mua quá sát giờ bắt đầu.",
        )
        live_cutoff_hours = st.slider(
            "Ngưỡng giờ tính là Live / Sát trận (giờ)",
            1.0,
            4.0,
            2.0,
            0.5,
            help="Các kèo có thời gian từ lúc mua (VWAP) đến lúc kết thúc dưới ngưỡng này bị tính là vốn đánh Live/sát giờ.",
        )
        max_live_cap_pct = st.slider("Trần % vốn vào Live / Sát trận", 0.0, 40.0, 15.0, 2.5)
        min_zero_sell_pct = st.slider("% Kèo giữ tuyệt đối không bán non", 40.0, 100.0, 75.0, 5.0)
        min_hold_redeem = st.slider("% Lãi giữ đáo hạn REDEEM", 50.0, 100.0, 75.0, 5.0)
        min_act_matched = st.number_input("Số kèo khớp Activity tối thiểu", min_value=5, max_value=100, value=10, step=5)

    with st.expander("📈 Lưới 3 & 4: True Edge & Rủi Ro", expanded=True):
        min_true_roi = st.slider("True ROI 1 chiều tối thiểu (%)", 0.0, 30.0, 4.0, 0.5)
        min_cap_edge = st.slider("TrueEdge tối thiểu (+%)", -2.0, 15.0, 1.0, 0.5)
        min_core_band = st.slider("% Vốn dải lõi 15%–90%", 40.0, 95.0, 65.0, 5.0)
        max_high_fav = st.slider("Trần vốn dải >=95¢ (%)", 0.0, 40.0, 15.0, 2.5)
        max_hedged_pct = st.slider("Trần % Vốn Hedge 2 đầu", 0.0, 30.0, 10.0, 2.0)
        require_ex_top5 = st.checkbox("Bắt buộc PnL ngoài Top 5 kèo > $0", value=True)
        max_act_pages = st.slider("Trang Activity tối đa (×500)", 2, 15, 5, 1)

    if st.session_state["wallet_cache"]:
        st.divider()
        st.caption(f"💾 Đang cache dữ liệu của **{len(st.session_state['wallet_cache'])}** ví.")
        if st.button("🧹 Xóa Cache API", use_container_width=True):
            st.session_state["wallet_cache"] = {}
            st.session_state["activity_cache"] = {}
            st.rerun()

step2_cfg = {
    "min_pv_pct": min_pv_pct,
    "min_net_pnl": min_net_pnl,
    "min_markets": min_markets,
    "max_markets": max_markets,
    "max_dead_open": max_dead_open,
    "max_inactive_days": max_inactive_days,
    "allow_live_open_override": allow_live_open_override,
}

step3_cfg = {
    "allow_3_of_4_pro": allow_3_of_4_pro,
    "min_hold_hours": min_hold_hours,
    "live_cutoff_hours": live_cutoff_hours,
    "max_live_cap_pct": max_live_cap_pct,
    "min_zero_sell_pct": min_zero_sell_pct,
    "min_hold_redeem": min_hold_redeem,
    "min_act_matched": min_act_matched,
    "min_true_roi": min_true_roi,
    "min_cap_edge": min_cap_edge,
    "min_core_band": min_core_band,
    "max_high_fav": max_high_fav,
    "max_hedged_pct": max_hedged_pct,
    "require_ex_top5": require_ex_top5,
    "max_act_pages": max_act_pages,
}

st.divider()

c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.2, 1.0])
selected_cat_label = c1.selectbox("Chọn Category", options=list(CATEGORY_OPTIONS.keys()), index=0)
selected_range_label = c2.selectbox("Chọn Range (Khung thời gian)", options=list(RANGE_OPTIONS.keys()), index=3)
selected_order = c3.selectbox("Xếp hạng theo", options=["PNL", "VOL"], index=0)
selected_top_n = c4.number_input("Số lượng ví (Top N)", min_value=10, max_value=100, value=50, step=10)

cat_code = CATEGORY_OPTIONS[selected_cat_label]
range_code = RANGE_OPTIONS[selected_range_label]

b1, b2 = st.columns([3, 1.4])
run_clicked = b1.button(
    f"🚀 QUÉT MỚI FULL 3 BƯỚC ({cat_code} • {range_code.upper()} • TOP {int(selected_top_n)})",
    type="primary",
    use_container_width=True,
)
recalc_clicked = b2.button(
    "⚡ Lọc lại từ Cache (< 0.1s)",
    use_container_width=True,
    disabled=st.session_state["df_step1_raw"].empty,
)

d1, d2 = st.columns([3, 1.4])
direct_wallet_input = d1.text_input(
    "🔍 Hoặc dán trực tiếp 1 địa chỉ ví bất kỳ để kiểm toán 4 Lưới (VD: ví CS2 mẫu):",
    value="",
    placeholder="0xc9bfc7986767173b4f1c3fc9b2c6b255fc7381c0",
)
direct_clicked = d2.button("🎯 Soi Trực Tiếp Ví Này", use_container_width=True)


def _execute_steps_2_3(df_s1: pd.DataFrame, meta: dict[str, Any]) -> None:
    pbar = st.progress(0.05, text="⏳ Đang chạy Bước 2: Thăm dò & cache sổ vị thế...")

    def cb_s2(done: int, total: int):
        pct = 0.05 + 0.50 * (done / max(1, total))
        pbar.progress(min(0.55, pct), text=f"⏳ Bước 2: Đã tải sổ vị thế {done}/{total} ví...")

    df_s2_passed, df_s2_all, s2_stats = step2_surface_filter(
        df_s1, cfg=step2_cfg, wallet_cache=st.session_state["wallet_cache"], progress_cb=cb_s2
    )

    pbar.progress(0.60, text=f"🔬 Bước 3A: Đang sàng lọc nhanh {len(df_s2_passed)} ví trên Positions cache...")

    def cb_s3(done: int, total: int):
        pct = 0.60 + 0.38 * (done / max(1, total))
        pbar.progress(min(0.98, pct), text=f"🔬 Bước 3B: Đang kéo Activity kiểm tra Live/Pre-match cho {done}/{total} ví...")

    df_final_pros, df_s3_all, details_map, s3_stats = step3_deep_ledger_audit(
        df_s2_passed,
        cfg=step3_cfg,
        wallet_cache=st.session_state["wallet_cache"],
        activity_cache=st.session_state["activity_cache"],
        progress_cb=cb_s3,
    )
    pbar.progress(1.0, text="✅ Hoàn tất kiểm toán 4 Lưới Lọc!")
    time.sleep(0.15)
    pbar.empty()

    meta["step2_stats"] = s2_stats
    meta["step3_stats"] = s3_stats

    st.session_state["df_final_pros"] = df_final_pros
    st.session_state["df_step3_all"] = df_s3_all
    st.session_state["df_step2_all"] = df_s2_all
    st.session_state["step3_details_map"] = details_map
    st.session_state["pipeline_meta"] = meta


if direct_clicked:
    clean_w = direct_wallet_input.strip().lower()
    if not EVM_ADDR_RE.match(clean_w):
        st.error("Địa chỉ ví không hợp lệ. Vui lòng nhập đúng định dạng `0x...` (42 ký tự).")
    else:
        df_direct = pd.DataFrame(
            [
                {
                    "rank": 0,
                    "user_name": f"Direct ({clean_w[:6]}...{clean_w[-4:]})",
                    "wallet": clean_w,
                    "pnl_usdc": 0.0,
                    "ui_volume_usdc": 0.0,
                    "pnl_to_ui_vol_pct": 0.0,
                    "x_username": "",
                    "verified": False,
                    "category": "DIRECT",
                    "time_period": "all",
                    "profile_url": f"https://polymarket.com/profile/{clean_w}",
                    "is_direct_lookup": True,
                }
            ]
        )
        meta_dir = {
            "fetched_at_vn": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S GMT+7"),
            "endpoint": "DIRECT_LOOKUP",
            "category": "DIRECT",
            "time_period": "all",
            "order_by": "DIRECT",
            "requested_top_n": 1,
            "returned_count": 1,
            "step1_elapsed_s": 0.0,
        }
        st.session_state["df_step1_raw"] = df_direct
        _execute_steps_2_3(df_direct, meta_dir)

elif run_clicked:
    t_start = time.perf_counter()
    with st.spinner(f"Đang thực thi Bước 1: Kéo Top {int(selected_top_n)} ví ({cat_code} • {range_code})..."):
        df_step1, meta = step1_fetch_leaderboard(
            category=cat_code,
            time_period=range_code,
            order_by=selected_order,
            top_n=int(selected_top_n),
        )
    meta["step1_elapsed_s"] = round(time.perf_counter() - t_start, 2)
    st.session_state["df_step1_raw"] = df_step1

    if df_step1.empty:
        st.session_state["df_step2_all"] = pd.DataFrame()
        st.session_state["df_step3_all"] = pd.DataFrame()
        st.session_state["df_final_pros"] = pd.DataFrame()
        st.session_state["pipeline_meta"] = meta
    else:
        _execute_steps_2_3(df_step1, meta)

elif recalc_clicked and not st.session_state["df_step1_raw"].empty:
    _execute_steps_2_3(st.session_state["df_step1_raw"], dict(st.session_state["pipeline_meta"]))

meta_res: dict[str, Any] = st.session_state["pipeline_meta"]
df_final: pd.DataFrame = st.session_state["df_final_pros"]
df_s3_all: pd.DataFrame = st.session_state["df_step3_all"]
df_s2_all: pd.DataFrame = st.session_state["df_step2_all"]
details_map: dict[str, pd.DataFrame] = st.session_state["step3_details_map"]

if meta_res:
    if df_s2_all.empty:
        st.error(
            f"Bước 1 không thu thập được dữ liệu ví nào từ `{meta_res.get('endpoint')}`.\n\n"
            f"- **Errors:** `{meta_res.get('errors')}`\n"
            f"- **Debug response:** `{meta_res.get('debug_notes')}`"
        )
    else:
        s2_stats = meta_res.get("step2_stats", {})
        s3_stats = meta_res.get("step3_stats", {})
        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bước 1 (Leaderboard)", f"{s2_stats.get('input_count', 0)} ví", f"{meta_res.get('step1_elapsed_s', 0)}s")
        m2.metric("Bước 2 (Đạt sàng lọc)", f"{s2_stats.get('passed_count', 0)} ví", f"{s2_stats.get('elapsed_s', 0)}s")
        m3.metric(
            "Bước 3B (Kiểm toán sâu)",
            f"{s3_stats.get('stage3b_count', 0)}/{s3_stats.get('input_count', 0)} ví",
            f"Dừng sớm {s3_stats.get('input_count', 0) - s3_stats.get('stage3b_count', 0)} ví âm ROI ở 3A",
        )
        m4.metric(
            "🏆 ĐẠT CHUẨN PRO",
            f"{s3_stats.get('passed_count', 0)} ví",
            f"B3: {s3_stats.get('elapsed_s', 0)}s • Sát chuẩn 3/4: {s3_stats.get('watch_count', 0)}",
        )

        s3_col_cfg = {
            "rank": st.column_config.NumberColumn("Hạng LB", format="#%d", width="small"),
            "s3_verdict": st.column_config.TextColumn("Kết luận B3", width="medium"),
            "gate_badge": st.column_config.TextColumn("Lưới đạt", width="small"),
            "user_name": st.column_config.TextColumn("Tên hiển thị", width="medium"),
            "wallet": st.column_config.TextColumn("Địa chỉ ví", width="medium"),
            "med_hold_h": st.column_config.NumberColumn("Hold TB", format="%.1fh"),
            "live_cap_pct": st.column_config.NumberColumn("% Vốn Live/Sát giờ", format="%.1f%%"),
            "med_clip_usdc": st.column_config.NumberColumn("Clip TB", format="$%,.0f"),
            "zero_sell_pct": st.column_config.NumberColumn("Zero-Sell", format="%.0f%%"),
            "hold_pnl_share_pct": st.column_config.NumberColumn("% Lãi REDEEM", format="%.0f%%"),
            "true_cost_usdc": st.column_config.NumberColumn("Vốn 1 chiều ($)", format="$%,.0f"),
            "ledger_pnl_usdc": st.column_config.NumberColumn("Net PnL sổ cái", format="$%+,.0f"),
            "true_roi_pct": st.column_config.NumberColumn("True ROI", format="%+.1f%%"),
            "cap_wr_pct": st.column_config.NumberColumn("ShareWR", format="%.1f%%"),
            "w_entry_pct": st.column_config.NumberColumn("BreakEven WR", format="%.1f%%"),
            "cap_edge_pct": st.column_config.NumberColumn("TrueEdge", format="%+.1f%%"),
            "core_band_pct": st.column_config.NumberColumn("% Vốn 15-90%", format="%.0f%%"),
            "hedged_cap_pct": st.column_config.NumberColumn("% Vốn Hedge", format="%.1f%%"),
            "pnl_ex_top5": st.column_config.NumberColumn("PnL ex-Top 5", format="$%+,.0f"),
            "act_match_info": st.column_config.TextColumn("Khớp Act", width="small"),
            "s3_flags": st.column_config.TextColumn("Chi tiết Lưới Rớt / Chẩn đoán", width="large"),
            "profile_url": st.column_config.LinkColumn("Polymarket", display_text="Mở Profile"),
        }
        s3_cols = [c for c in s3_col_cfg.keys() if c in df_s3_all.columns]

        st.subheader(f"🏆 Bảng Kiểm Toán 4 Lưới Lọc Bước 3 ({len(df_s3_all)} ví đã qua Bước 2)")
        if df_s3_all.empty:
            st.warning("Không có ví nào vượt qua Bước 2 để đưa vào Bước 3.")
        else:
            st.dataframe(df_s3_all[s3_cols], use_container_width=True, hide_index=True, column_config=s3_col_cfg)

        # ==================== TRÌNH SOI SÂU X-RAY ====================
        if not df_s3_all.empty and details_map:
            st.divider()
            st.subheader("🔬 Soi Sâu X-Ray Từng Ví (Đọc tức thì từ Cache)")

            label_to_wallet: dict[str, str] = {}
            for _, r in df_s3_all.iterrows():
                w = str(r["wallet"]).strip().lower()
                if w in details_map:
                    lbl = f"#{int(r['rank'])} | {r['s3_verdict']} ({r['gate_badge']}) — {r['user_name']} ({w[:8]}...)"
                    label_to_wallet[lbl] = w

            if label_to_wallet:
                picked_lbl = st.selectbox("Chọn ví cần giải phẫu chi tiết:", options=list(label_to_wallet.keys()))
                picked_w = label_to_wallet[picked_lbl]
                pos_x = details_map[picked_w].copy()
                tot_c = float(pos_x["true_cost"].sum())

                x_left, x_right = st.columns(2)
                with x_left:
                    st.markdown("**1. Phân rã Phong cách Thoát lệnh (`exit_style`)**")
                    ex_rows = []
                    for st_n, g in pos_x.groupby("exit_style"):
                        gc = float(g["true_cost"].sum())
                        gp = float(g["total_pnl"].sum())
                        ex_rows.append(
                            {
                                "Phong cách": st_n,
                                "Số vị thế": len(g),
                                "Win Rate": f"{g['real_win'].mean() * 100:.1f}%",
                                "Vốn ($)": round(gc, 1),
                                "% Vốn": f"{gc / tot_c * 100 if tot_c > 0 else 0:.1f}%",
                                "PnL ($)": round(gp, 1),
                                "ROI": f"{gp / gc * 100 if gc > 0 else 0:+.1f}%",
                            }
                        )
                    st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)

                with x_right:
                    st.markdown("**2. Phân bổ Vốn & ROI theo Dải Xác suất**")
                    bins = [0.0, 0.10, 0.15, 0.50, 0.75, 0.90, 0.95, 1.01]
                    labels = ["<10% (Penny)", "10–15%", "15–50%", "50–75%", "75–90%", "90–95%", ">=95%"]
                    pos_x["prob_band"] = pd.cut(pos_x["avg_price"], bins=bins, labels=labels, right=False)
                    b_rows = []
                    for b_n, g in pos_x.groupby("prob_band", observed=False):
                        if len(g) == 0:
                            continue
                        gc = float(g["true_cost"].sum())
                        gp = float(g["total_pnl"].sum())
                        b_rows.append(
                            {
                                "Dải giá": str(b_n),
                                "Số vị thế": len(g),
                                "Win Rate": f"{g['real_win'].mean() * 100:.1f}%",
                                "Vốn ($)": round(gc, 1),
                                "% Vốn": f"{gc / tot_c * 100 if tot_c > 0 else 0:.1f}%",
                                "PnL ($)": round(gp, 1),
                                "ROI": f"{gp / gc * 100 if gc > 0 else 0:+.1f}%",
                            }
                        )
                    st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

                x_bot_l, x_bot_r = st.columns(2)
                with x_bot_l:
                    st.markdown("**3. Kỷ luật đi tiền theo Quy mô kèo (Conviction Sizing)**")
                    q50 = pos_x["true_cost"].quantile(0.50)
                    q80 = pos_x["true_cost"].quantile(0.80)
                    pos_x["size_tier"] = np.where(
                        pos_x["true_cost"] >= q80,
                        f"Top 20% Kèo Lớn (>= ${q80:,.1f})",
                        np.where(
                            pos_x["true_cost"] > q50,
                            f"Nhóm Trung Bình (${q50:,.1f} – ${q80:,.1f})",
                            f"Bottom 50% Kèo Nhỏ (<= ${q50:,.1f})",
                        ),
                    )
                    t_rows = []
                    for t_n, g in pos_x.groupby("size_tier"):
                        gc = float(g["true_cost"].sum())
                        gp = float(g["total_pnl"].sum())
                        t_rows.append(
                            {
                                "Phân khúc quy mô": t_n,
                                "Số vị thế": len(g),
                                "Win Rate": f"{g['real_win'].mean() * 100:.1f}%",
                                "Tổng vốn ($)": round(gc, 1),
                                "PnL ($)": round(gp, 1),
                                "ROI": f"{gp / gc * 100 if gc > 0 else 0:+.1f}%",
                            }
                        )
                    st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

                with x_bot_r:
                    st.markdown("**4. Giải phẫu Nhịp Vào Lệnh (Pre-match Sớm vs. Sát giờ/Live)**")
                    valid_h = pos_x[pos_x["hold_hours"].notna()].copy()
                    if valid_h.empty:
                        st.info("Ví này chưa có đủ dữ liệu thời gian Hold khớp Activity.")
                    else:
                        h_bins = [0.0, 2.0, 4.0, 12.0, 1e9]
                        h_labels = ["< 2.0h (Live / Sát giờ)", "2.0h – 4h (Cận trận)", "4h – 12h (Pre-match)", "> 12h (Vào sớm)"]
                        valid_h["hold_tier"] = pd.cut(valid_h["hold_hours"], bins=h_bins, labels=h_labels, right=False)
                        h_rows = []
                        for h_n, g in valid_h.groupby("hold_tier", observed=False):
                            if len(g) == 0:
                                continue
                            gc = float(g["true_cost"].sum())
                            gp = float(g["total_pnl"].sum())
                            h_rows.append(
                                {
                                    "Khung giữ lệnh": str(h_n),
                                    "Số kèo": len(g),
                                    "Vốn ($)": round(gc, 1),
                                    "% Vốn": f"{gc / tot_c * 100 if tot_c > 0 else 0:.1f}%",
                                    "PnL ($)": round(gp, 1),
                                    "Số lệnh mua TB": f"{g['buy_count'].mean():.1f}",
                                    "Clip TB ($)": f"${g['avg_clip_usdc'].median():,.0f}",
                                }
                            )
                        st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

                export_pos_cols = [
                    "entry_time_vn", "status", "exit_style", "condition_id", "token_id",
                    "title", "outcome", "is_hedged", "avg_price", "true_cost", "total_pnl",
                    "roi_pct", "real_win", "buy_count", "sell_count", "avg_clip_usdc",
                    "buy_price_spread_c", "accum_mins", "hold_hours", "url",
                ]
                avail_ex = [c for c in export_pos_cols if c in pos_x.columns]
                st.download_button(
                    f"📥 Tải lịch sử sổ cái sạch của `{picked_w[:10]}...` (CSV)",
                    data=pos_x[avail_ex].to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"xray_{picked_w[:10]}.csv",
                    mime="text/csv",
                )

        # ==================== BẢNG PHỄU BƯỚC 2 ====================
        s2_col_cfg = {
            "rank": st.column_config.NumberColumn("Hạng LB", format="#%d", width="small"),
            "step2_status": st.column_config.TextColumn("Trạng thái B2", width="medium"),
            "user_name": st.column_config.TextColumn("Tên hiển thị", width="medium"),
            "wallet": st.column_config.TextColumn("Địa chỉ ví", width="medium"),
            "pnl_usdc": st.column_config.NumberColumn("PnL ($)", format="$%+,.2f"),
            "ui_volume_usdc": st.column_config.NumberColumn("UI Volume ($)", format="$%,.2f"),
            "pnl_to_ui_vol_pct": st.column_config.NumberColumn("P/V (%)", format="%.2f%%"),
            "settled_markets": st.column_config.TextColumn("Kèo đã đóng", width="small"),
            "open_summary": st.column_config.TextColumn("OPEN (Sống / Xác 0¢)", width="medium"),
            "recent_30d": st.column_config.TextColumn("Hoạt động 30d", width="small"),
            "last_active": st.column_config.TextColumn("Lệnh cuối", width="small"),
            "step2_reason": st.column_config.TextColumn("Lý do Đạt / Loại B2", width="large"),
        }
        s2_cols = [c for c in s2_col_cfg.keys() if c in df_s2_all.columns]
        with st.expander(f"📋 Xem lại Phễu Bước 2 toàn bộ {len(df_s2_all)} ví ban đầu", expanded=False):
            st.dataframe(df_s2_all[s2_cols], use_container_width=True, hide_index=True, column_config=s2_col_cfg)

        dl1, dl2 = st.columns(2)
        prefix = f"step3_{meta_res['category'].lower()}_{meta_res['time_period'].lower()}"
        dl1.download_button(
            f"📥 Tải {len(df_final)} ví CHUẨN PRO (CSV)",
            data=df_final.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{prefix}_pros.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=df_final.empty,
        )
        dl2.download_button(
            f"📥 Tải toàn bộ bảng kiểm toán Bước 3 ({len(df_s3_all)} ví)",
            data=df_s3_all.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{prefix}_audited_all.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=df_s3_all.empty,
        )
