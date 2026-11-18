from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import streamlit as st

st.set_page_config(
    page_title="Polymarket Wallet Finder — Kiểm Toán Bước 2",
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
    """Mỗi worker thread giữ riêng 1 Session Keep-Alive để tối ưu tốc độ TLS."""
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Polymarket-Wallet-Finder/2.3",
            }
        )
        _THREAD_LOCAL.session = sess
    return sess


def request_json_with_retry(
    url: str,
    params: dict[str, Any],
    retries: int = 3,
    timeout: int = 12,
) -> Any:
    session = get_http_session()
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Gọi {url} thất bại với params={params}: {last_err}")


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


def _fetch_positions_by_status(
    wallet: str,
    status_val: str,
    max_pages: int = 8,
    stop_cid_cap: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Kéo danh sách vị thế theo đúng status trên Data API v2 (cursor pagination)."""
    rows: list[dict[str, Any]] = []
    unique_cids: set[str] = set()
    cursor: str | None = None
    capped = False

    for _ in range(max_pages):
        params: dict[str, Any] = {"user": wallet, "status": status_val, "limit": 500}
        if cursor:
            params["cursor"] = cursor
        try:
            payload = request_json_with_retry(POSITIONS_V2_URL, params=params, retries=2, timeout=10)
        except Exception:
            break

        if isinstance(payload, dict):
            batch = payload.get("data") or []
            pagination = payload.get("pagination") or {}
            cursor = pagination.get("next_cursor")
            has_more = bool(pagination.get("has_more", bool(cursor)))
        elif isinstance(payload, list):
            batch = payload
            cursor = None
            has_more = False
        else:
            break

        for r in batch:
            if not isinstance(r, dict):
                continue
            rows.append(r)
            cid = str(r.get("condition_id") or r.get("conditionId") or "").strip()
            if cid:
                unique_cids.add(cid)

        if stop_cid_cap is not None and len(unique_cids) > stop_cid_cap:
            capped = True
            break
        if not cursor or not has_more:
            break

    return rows, capped


# ==============================================================================
# BƯỚC 1 (HOÀN THÀNH): THU THẬP DỮ LIỆU THÔ TOP N VÍ TỪ LEADERBOARD
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
    used_endpoint = LEADERBOARD_V2_URL
    errors: list[str] = []
    max_pages = max(1, math.ceil(top_n / 50))

    cursor: str | None = None
    v2_failed_first_page = False

    for page_idx in range(max_pages):
        if len(raw_records) >= top_n:
            break

        if cursor is None:
            params: dict[str, Any] = {
                "category": cat_code,
                "time_period": tp_code,
                "order_by": ob_code,
                "limit": min(50, top_n - len(raw_records)),
            }
        else:
            params = {"cursor": cursor}

        try:
            payload = request_json_with_retry(LEADERBOARD_V2_URL, params=params)
        except Exception as exc:
            errors.append(f"[v2] {exc}")
            if page_idx == 0:
                v2_failed_first_page = True
            break

        if isinstance(payload, dict):
            batch = payload.get("data") or payload.get("results") or []
            pagination = payload.get("pagination") or {}
            cursor = pagination.get("next_cursor")
            has_more = bool(pagination.get("has_more", bool(cursor)))
        elif isinstance(payload, list):
            batch = payload
            cursor = None
            has_more = False
        else:
            break

        if not batch:
            break

        raw_records.extend(r for r in batch if isinstance(r, dict))

        if not cursor or not has_more:
            break

    if v2_failed_first_page and not raw_records:
        used_endpoint = LEADERBOARD_V1_URL
        offset = 0
        while len(raw_records) < top_n:
            batch_limit = min(50, top_n - len(raw_records))
            try:
                payload_v1 = request_json_with_retry(
                    LEADERBOARD_V1_URL,
                    params={
                        "category": cat_code,
                        "timePeriod": tp_code.upper(),
                        "orderBy": ob_code,
                        "limit": batch_limit,
                        "offset": offset,
                    },
                )
                batch_v1 = (
                    payload_v1.get("data") or []
                    if isinstance(payload_v1, dict)
                    else (payload_v1 or [])
                )
                if not batch_v1:
                    break
                raw_records.extend(r for r in batch_v1 if isinstance(r, dict))
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

        x_user = _pick_first_str(item, ("x_username", "xUsername", "twitter", "x_handle"))
        verified_flag = bool(
            item.get("verified_badge")
            or item.get("verifiedBadge")
            or item.get("verified")
            or False
        )

        normalized_rows.append(
            {
                "rank": rank_val,
                "user_name": user_name,
                "wallet": wallet if wallet else "",
                "pnl_usdc": round(pnl_val, 2),
                "ui_volume_usdc": round(vol_val, 2),
                "pnl_to_ui_vol_pct": round(pnl_vol_pct, 2),
                "x_username": x_user,
                "verified": verified_flag,
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
# BƯỚC 2 (ĐANG KIỂM TOÁN): LỌC BỀ MẶT, CHẶN XÁC VỊ THẾ (0¢) & KHÓA ĐỘ TƯƠI
# ==============================================================================

def _probe_wallet_surface_v2(wallet: str, max_settled_cap: int) -> dict[str, Any]:
    """
    Thăm dò chuẩn xác 1 ví qua Data API v2:
      1. Activity (/v2/activity?limit=100): Độ tươi (days_since_last), số kèo hoạt động 30d qua.
      2. OPEN (/v2/positions?status=OPEN): Tách riêng lệnh OPEN đang sống vs "xác vị thế" về giá 0¢.
      3. SETTLED (/v2/positions?status=CLOSED + REDEEMABLE + REDEEMABLE_LOST): Đếm đúng số kèo đã đánh.
    """
    now_utc = datetime.now(timezone.utc)
    cutoff_30d = now_utc - timedelta(days=30)

    days_since_last: float | None = None
    recent_market_title = ""
    recent_30d_cids: set[str] = set()

    # 1. Kiểm tra lịch sử hoạt động gần nhất (/v2/activity?user=...&limit=100)
    try:
        act_payload = request_json_with_retry(
            ACTIVITY_V2_URL, params={"user": wallet, "limit": 100}, retries=2, timeout=10
        )
        act_rows = act_payload.get("data") or [] if isinstance(act_payload, dict) else (act_payload or [])
        latest_dt: datetime | None = None
        for r in act_rows:
            if not isinstance(r, dict):
                continue
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

    # 2. Kiểm tra vị thế đang mở (status=OPEN): tách lệnh sống vs "xác vị thế" về giá 0¢
    open_rows, open_capped = _fetch_positions_by_status(wallet, "OPEN", max_pages=2, stop_cid_cap=500)
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
        elif cur_v > 1.0 or (0.01 < cur_p < 0.99 and cur_s > 0):
            live_open_cnt += 1
        else:
            dead_zero_open_cnt += 1

    # 3. Đếm số kèo đã giao dịch & ngã ngũ (CLOSED + REDEEMABLE + REDEEMABLE_LOST)
    settled_cids: set[str] = set()
    max_pages_closed = max(2, math.ceil((max_settled_cap + 200) / 500))
    closed_rows, settled_capped = _fetch_positions_by_status(
        wallet, "CLOSED", max_pages=max_pages_closed, stop_cid_cap=max_settled_cap
    )
    for r in closed_rows:
        cid = str(r.get("condition_id") or r.get("conditionId") or "").strip()
        if cid:
            settled_cids.add(cid)
        if not recent_market_title:
            recent_market_title = str(r.get("title") or "").strip()

    if not settled_capped:
        for extra_st in ("REDEEMABLE", "REDEEMABLE_LOST"):
            extra_rows, _ = _fetch_positions_by_status(wallet, extra_st, max_pages=2, stop_cid_cap=500)
            for r in extra_rows:
                cid = str(r.get("condition_id") or r.get("conditionId") or "").strip()
                if cid:
                    settled_cids.add(cid)

    settled_count = len(settled_cids)
    if settled_count > max_settled_cap:
        settled_capped = True

    return {
        "settled_markets": settled_count,
        "settled_capped": settled_capped,
        "live_open": live_open_cnt,
        "dead_zero_open": dead_zero_open_cnt,
        "open_capped": open_capped,
        "recent_30d_mkts": len(recent_30d_cids),
        "days_since_last": round(days_since_last, 1) if days_since_last is not None else None,
        "recent_market": recent_market_title[:75] if recent_market_title else "N/A",
    }


def step2_surface_filter(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t0 = time.perf_counter()
    if df.empty:
        return df.copy(), df.copy(), {"elapsed_s": 0.0, "input_count": 0, "passed_count": 0}

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
        reasons = []
        if not w or not w.startswith("0x"):
            reasons.append("Thiếu địa chỉ ví 0x")
        if float(row["pnl_usdc"]) < min_pnl:
            reasons.append(f"PnL < ${min_pnl:,.0f}")
        if float(row["pnl_to_ui_vol_pct"]) < min_pv:
            reasons.append(f"P/V {row['pnl_to_ui_vol_pct']:.2f}% < {min_pv:.1f}%")

        if reasons:
            pre_fail_reasons[w] = reasons
        else:
            pre_pass_wallets.append(w)

    probe_map: dict[str, dict[str, Any]] = {}
    if pre_pass_wallets:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_wallet = {
                executor.submit(_probe_wallet_surface_v2, w, max_mkts): w
                for w in pre_pass_wallets
            }
            for fut in as_completed(future_to_wallet):
                w = future_to_wallet[fut]
                try:
                    probe_map[w] = fut.result()
                except Exception as exc:
                    probe_map[w] = {
                        "settled_markets": 0,
                        "settled_capped": False,
                        "live_open": 0,
                        "dead_zero_open": 0,
                        "open_capped": False,
                        "recent_30d_mkts": 0,
                        "days_since_last": None,
                        "recent_market": f"Lỗi: {exc}",
                    }

    status_col: list[str] = []
    passed_flags: list[bool] = []
    settled_col: list[str] = []
    open_summary_col: list[str] = []
    recent_30d_col: list[str] = []
    fresh_col: list[str] = []
    recent_col: list[str] = []
    reason_col: list[str] = []

    for _, row in work_df.iterrows():
        w = str(row["wallet"]).strip().lower()
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

        info = probe_map.get(w, {})
        m_cnt = int(info.get("settled_markets") or 0)
        m_capped = bool(info.get("settled_capped", False))
        live_open = int(info.get("live_open") or 0)
        dead_open = int(info.get("dead_zero_open") or 0)
        open_capped = bool(info.get("open_capped", False))
        r30_cnt = int(info.get("recent_30d_mkts") or 0)
        d_last = info.get("days_since_last")

        reasons = []
        if m_cnt < min_mkts:
            reasons.append(f"Ít mẫu đã đóng ({m_cnt} < {min_mkts} kèo)")
        elif m_capped or m_cnt > max_mkts:
            reasons.append(f"Quá trần kèo đã đóng (>{max_mkts})")

        if open_capped or dead_open > max_dead_open:
            dead_disp = f">{max_dead_open}" if open_capped else str(dead_open)
            reasons.append(f"Ôm xác vị thế về 0 ({dead_disp} lệnh $0 > {max_dead_open})")

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
            reason_col.append("Sạch xác vị thế, đủ kèo & đang chạy")
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

    stats_log = {
        "elapsed_s": elapsed,
        "input_count": len(work_df),
        "pre_pass_count": len(pre_pass_wallets),
        "passed_count": len(df_passed),
    }
    return df_passed, work_df, stats_log


# ==============================================================================
# BƯỚC 3 (CHƯA TRIỂN KHAI — CHỜ NGHIỆM THU XONG BƯỚC 2)
# ==============================================================================

def step3_deep_ledger_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    [PLACEHOLDER - CHƯA TRIỂN KHAI]
    Giữ nguyên 100% danh sách ví đã đạt Bước 2 cho đến khi bạn xác nhận xong Bước 2.
    """
    return df.copy(), "Chưa triển khai (Pass-through 100%)"


# ==============================================================================
# GIAO DIỆN STREAMLIT ĐIỀU KHIỂN PIPELINE (BƯỚC 1 -> BƯỚC 2)
# ==============================================================================

st.title("📡 Polymarket Wallet Finder — Kiểm Toán Bước 2")
st.caption(
    "**Bước 1:** Thu thập Top N ví Leaderboard (Xong) → "
    "**Bước 2:** Lọc P/V, chặn ôm xác vị thế 0¢ & khóa độ tươi (Đang kiểm toán) → "
    "*Bước 3: Kiểm toán sổ cái sâu (Chờ)*."
)

s1, s2, s3 = st.columns(3)
s1.success("✅ **Bước 1: Thu thập Top N ví**\n\nHoạt động (< 1 giây)")
s2.success("✅ **Bước 2: Lọc Bề mặt & Chặn Xác 0¢**\n\nĐang kiểm toán (CLOSED vs xác 0¢)")
s3.info("⏳ **Bước 3: Kiểm toán Sổ cái Sâu**\n\nChưa triển khai (Pass-through)")

with st.sidebar:
    st.header("⚙️ Ngưỡng Lọc Bước 2")
    st.caption("Tinh chỉnh các chốt chặn của Bước 2 tại đây:")
    min_pv_pct = st.slider("1. Sàn PnL / UI Volume (%)", 0.0, 15.0, 4.0, 0.5)
    min_net_pnl = st.number_input("2. Net PnL tối thiểu ($)", min_value=0.0, value=5000.0, step=1000.0)

    c_m1, c_m2 = st.columns(2)
    min_markets = c_m1.number_input("3. Kèo đóng tối thiểu", min_value=1, max_value=500, value=80, step=10)
    max_markets = c_m2.number_input("4. Kèo đóng tối đa", min_value=50, max_value=10000, value=3500, step=250)

    max_dead_open = st.number_input(
        "5. Trần 'xác vị thế' về 0¢ tối đa",
        min_value=0,
        max_value=500,
        value=15,
        step=5,
        help="Chặn bot rải lệnh & ví ôm xác vị thế 0¢ chưa đóng trong tab OPEN.",
    )
    max_inactive_days = st.slider("6. Ngừng chạy tối đa (ngày)", 1, 90, 30, 1)
    allow_live_open_override = st.checkbox("Giữ ví nếu đang có lệnh OPEN sống (> $1)", value=True)

step2_cfg = {
    "min_pv_pct": min_pv_pct,
    "min_net_pnl": min_net_pnl,
    "min_markets": min_markets,
    "max_markets": max_markets,
    "max_dead_open": max_dead_open,
    "max_inactive_days": max_inactive_days,
    "allow_live_open_override": allow_live_open_override,
}

st.divider()

c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.2, 1.0])

selected_cat_label = c1.selectbox(
    "Chọn Category",
    options=list(CATEGORY_OPTIONS.keys()),
    index=0,
)
selected_range_label = c2.selectbox(
    "Chọn Range (Khung thời gian)",
    options=list(RANGE_OPTIONS.keys()),
    index=2,
)
selected_order = c3.selectbox(
    "Xếp hạng theo",
    options=["PNL", "VOL"],
    index=0,
)
selected_top_n = c4.number_input(
    "Số lượng ví (Top N)",
    min_value=10,
    max_value=100,
    value=40,
    step=10,
)

cat_code = CATEGORY_OPTIONS[selected_cat_label]
range_code = RANGE_OPTIONS[selected_range_label]

run_clicked = st.button(
    f"🚀 CHẠY BƯỚC 1 & BƯỚC 2 ({cat_code} • {range_code.upper()} • TOP {int(selected_top_n)})",
    type="primary",
    use_container_width=True,
)

for key, default_val in [
    ("df_passed", pd.DataFrame()),
    ("df_all_step2", pd.DataFrame()),
    ("pipeline_meta", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

if run_clicked:
    with st.spinner(f"Đang kéo Top {int(selected_top_n)} ví & kiểm toán Bước 2 ({cat_code})..."):
        t_start = time.perf_counter()
        df_step1, meta = step1_fetch_leaderboard(
            category=cat_code,
            time_period=range_code,
            order_by=selected_order,
            top_n=int(selected_top_n),
        )
        t_step1 = round(time.perf_counter() - t_start, 2)

        df_step2_passed, df_step2_all, s2_stats = step2_surface_filter(df_step1, cfg=step2_cfg)
        df_step3, status_step3 = step3_deep_ledger_audit(df_step2_passed)

        meta["step1_elapsed_s"] = t_step1
        meta["step2_stats"] = s2_stats
        meta["step3_status"] = status_step3

        st.session_state["df_passed"] = df_step3
        st.session_state["df_all_step2"] = df_step2_all
        st.session_state["pipeline_meta"] = meta

meta_res: dict[str, Any] = st.session_state["pipeline_meta"]
df_passed: pd.DataFrame = st.session_state["df_passed"]
df_all_s2: pd.DataFrame = st.session_state["df_all_step2"]

if meta_res:
    if df_all_s2.empty:
        st.error(f"Không thu thập được dữ liệu từ API. Chi tiết lỗi: {meta_res.get('errors')}")
    else:
        s2_stats = meta_res.get("step2_stats", {})
        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Bước 1 (Thu thập thô)",
            f"{s2_stats.get('input_count', 0)} ví",
            f"{meta_res.get('step1_elapsed_s', 0)}s",
        )
        m2.metric(
            "Qua chốt P/V & PnL",
            f"{s2_stats.get('pre_pass_count', 0)} ví",
            f"Sàn P/V >= {min_pv_pct:.1f}%",
        )
        m3.metric(
            "ĐẠT BƯỚC 2 (Chờ Bước 3)",
            f"{s2_stats.get('passed_count', 0)} ví",
            f"Lọc mất {s2_stats.get('elapsed_s', 0)}s",
        )
        m4.metric(
            "Bước 3 (Kiểm toán sâu)",
            "Chưa triển khai",
            "Pass-through",
        )

        col_cfg = {
            "rank": st.column_config.NumberColumn("Hạng", format="#%d", width="small"),
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
            "recent_market": st.column_config.TextColumn("Kèo gần nhất", width="large"),
            "step2_reason": st.column_config.TextColumn("Lý do Đạt / Loại", width="large"),
            "profile_url": st.column_config.LinkColumn("Polymarket", display_text="Mở Profile"),
        }

        display_cols = [c for c in col_cfg.keys() if c in df_all_s2.columns]

        st.subheader(f"✅ Danh sách {len(df_passed)} ví ĐẠT BƯỚC 2 (Sạch xác vị thế & Đang chạy)")
        st.dataframe(
            df_passed[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
        )

        df_rejected = df_all_s2[~df_all_s2["step2_passed"]].reset_index(drop=True)
        if not df_rejected.empty:
            with st.expander(f"🔎 Xem chi tiết {len(df_rejected)} ví BỊ LOẠI ở Bước 2 & Lý do cụ thể", expanded=True):
                st.dataframe(
                    df_rejected[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config=col_cfg,
                )

        dl1, dl2 = st.columns(2)
        prefix = f"step2_{meta_res['category'].lower()}_{meta_res['time_period'].lower()}"

        dl1.download_button(
            f"📥 Tải {len(df_passed)} ví ĐẠT BƯỚC 2 (CSV)",
            data=df_passed.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{prefix}_passed.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=df_passed.empty,
        )
        dl2.download_button(
            f"📥 Tải báo cáo đầy đủ {len(df_all_s2)} ví (CSV)",
            data=df_all_s2.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{prefix}_all_audited.csv",
            mime="text/csv",
            use_container_width=True,
        )
