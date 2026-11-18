import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import streamlit as st

st.set_page_config(
    page_title="Polymarket Wallet Finder — Step 1 Seed Collector",
    page_icon="📡",
    layout="wide",
)

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & HẰNG SỐ API
# ==============================================================================

LEADERBOARD_V2_URL = "https://data-api.polymarket.com/v2/leaderboard"
LEADERBOARD_V1_URL = "https://data-api.polymarket.com/v1/leaderboard"
VN_TZ = timezone(timedelta(hours=7))
EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# Đủ 11 danh mục khớp 100% với dropdown trên polymarket.com/leaderboard
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


def build_http_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "Polymarket-Wallet-Finder/1.1",
        }
    )
    return session


def request_json_with_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    retries: int = 3,
    timeout: int = 12,
) -> Any:
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(1.2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Gọi {url} thất bại với params={params}: {last_err}")


def _extract_wallet_address(item: dict[str, Any]) -> str:
    """Tự động dò địa chỉ ví 0x... kể cả khi API đổi tên trường hoặc lồng object."""
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
        if isinstance(val, str):
            s = val.strip()
            if EVM_ADDR_RE.match(s):
                return s.lower()
        elif isinstance(val, dict):
            nested = _extract_wallet_address(val)
            if nested:
                return nested

    # Quét toàn bộ giá trị chuỗi trong record nếu khóa nằm ngoài danh sách trên
    for val in item.values():
        if isinstance(val, str):
            s = val.strip()
            if EVM_ADDR_RE.match(s):
                return s.lower()
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


# ==============================================================================
# BƯỚC 1 (ĐÃ TRIỂN KHAI): THU THẬP DỮ LIỆU THÔ TOP N VÍ TỪ LEADERBOARD
# ==============================================================================

def step1_fetch_leaderboard(
    category: str,
    time_period: str,
    order_by: str = "PNL",
    top_n: int = 40,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Kéo nguyên bản Top N ví từ Polymarket Leaderboard (/v2/leaderboard theo cursor,
    tự động fallback sang /v1/leaderboard nếu route v2 gặp lỗi tham số).
    """
    session = build_http_session()
    cat_code = category.strip().upper()
    tp_code = time_period.strip().lower()
    ob_code = order_by.strip().upper()

    raw_records: list[dict[str, Any]] = []
    used_endpoint = LEADERBOARD_V2_URL
    errors: list[str] = []
    max_pages = max(1, math.ceil(top_n / 50))

    # 1. Ưu tiên gọi Data API v2 (phân trang bằng cursor)
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
            payload = request_json_with_retry(session, LEADERBOARD_V2_URL, params=params)
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

    # 2. Fallback sang /v1/leaderboard nếu v2 lỗi ngay trang đầu
    if v2_failed_first_page and not raw_records:
        used_endpoint = LEADERBOARD_V1_URL
        offset = 0
        while len(raw_records) < top_n:
            batch_limit = min(50, top_n - len(raw_records))
            try:
                payload_v1 = request_json_with_retry(
                    session,
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
                "wallet": wallet if wallet else "N/A (Xem Raw JSON)",
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
# BƯỚC 2 (CHƯA TRIỂN KHAI): LỌC BỀ MẶT & ĐỘ TƯƠI HOẠT ĐỘNG
# ==============================================================================

def step2_surface_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    [PLACEHOLDER - CHƯA TRIỂN KHAI]
    Sau này sẽ bổ sung logic kiểm tra độ tươi (last_trade <= 14 ngày),
    số lượng kèo (80..1500) và tỷ trọng kèo đúng ngách.
    Hiện tại giữ nguyên 100% dữ liệu từ Bước 1.
    """
    return df.copy(), "Chưa triển khai (Pass-through 100%)"


# ==============================================================================
# BƯỚC 3 (CHƯA TRIỂN KHAI): KIỂM TOÁN SỔ CÁI CHUYÊN SÂU (/positions + /activity)
# ==============================================================================

def step3_deep_ledger_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    [PLACEHOLDER - CHƯA TRIỂN KHAI]
    Sau này sẽ bổ sung logic đối chiếu /v2/positions & /v2/activity
    (True ROI 1 chiều, CapEdge, Z-score, REDEEM >= 80%, PnL ex-Top 5 > 0).
    Hiện tại giữ nguyên 100% dữ liệu từ Bước 2.
    """
    return df.copy(), "Chưa triển khai (Pass-through 100%)"


# ==============================================================================
# GIAO DIỆN STREAMLIT ĐIỀU KHIỂN PIPELINE
# ==============================================================================

st.title("📡 Polymarket Wallet Finder — Khung Nền Dự Án (Bước 1)")
st.caption(
    "Lộ trình 3 bước trong cùng 1 ứng dụng: "
    "**Bước 1: Thu thập Top N ví thô (Đang chạy)** → "
    "*Bước 2: Lọc bề mặt & độ tươi (Chờ)* → "
    "*Bước 3: Kiểm toán sổ cái sâu (Chờ)*."
)

s1, s2, s3 = st.columns(3)
s1.success("✅ **Bước 1: Thu thập Top N ví**\n\nĐang hoạt động (Dữ liệu gốc 100%)")
s2.info("⏳ **Bước 2: Lọc bề mặt & độ tươi**\n\nChưa triển khai (Pass-through)")
s3.info("⏳ **Bước 3: Kiểm toán sổ cái sâu**\n\nChưa triển khai (Pass-through)")

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
    f"📥 KÉO TOP {int(selected_top_n)} VÍ HÀNG ĐẦU ({cat_code} • {range_code.upper()})",
    type="primary",
    use_container_width=True,
)

if "pipeline_df" not in st.session_state:
    st.session_state["pipeline_df"] = pd.DataFrame()
if "pipeline_meta" not in st.session_state:
    st.session_state["pipeline_meta"] = {}

if run_clicked:
    with st.spinner(f"Đang thực thi Bước 1: Kéo Top {int(selected_top_n)} ví từ `{cat_code}` ({range_code})..."):
        df_step1, meta = step1_fetch_leaderboard(
            category=cat_code,
            time_period=range_code,
            order_by=selected_order,
            top_n=int(selected_top_n),
        )
        df_step2, status_step2 = step2_surface_filter(df_step1)
        df_step3, status_step3 = step3_deep_ledger_audit(df_step2)

        meta["step2_status"] = status_step2
        meta["step3_status"] = status_step3

        st.session_state["pipeline_df"] = df_step3
        st.session_state["pipeline_meta"] = meta

df_result: pd.DataFrame = st.session_state["pipeline_df"]
meta_result: dict[str, Any] = st.session_state["pipeline_meta"]

if meta_result:
    if df_result.empty:
        st.error(
            f"Không chuẩn hóa được dữ liệu ví nào. Chi tiết lỗi API: {meta_result.get('errors')}"
        )
    else:
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Số ví thu thập (Bước 1)", f"{len(df_result)} / {meta_result['requested_top_n']} ví")
        m2.metric("Category • Range", f"{meta_result['category']} • {meta_result['time_period'].upper()}")
        m3.metric("Tổng PnL Top nhóm", f"${df_result['pnl_usdc'].sum():+,.0f}")
        m4.metric("Bước 2 & Bước 3", "Chưa triển khai")

        st.caption(
            f"Endpoint: `{meta_result['endpoint']}` • "
            f"Thời điểm kéo: `{meta_result['fetched_at_vn']}` • "
            f"Trạng thái lọc: Chưa áp dụng (Giữ nguyên 100% Top {len(df_result)} ví)"
        )

        st.dataframe(
            df_result,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Hạng", format="#%d", width="small"),
                "user_name": st.column_config.TextColumn("Tên hiển thị", width="medium"),
                "wallet": st.column_config.TextColumn("Địa chỉ ví (Proxy Wallet)", width="large"),
                "pnl_usdc": st.column_config.NumberColumn("PnL ($)", format="$%+,.2f"),
                "ui_volume_usdc": st.column_config.NumberColumn("UI Volume ($)", format="$%,.2f"),
                "pnl_to_ui_vol_pct": st.column_config.NumberColumn("PnL / UI Vol (%)", format="%.2f%%"),
                "x_username": st.column_config.TextColumn("X (Twitter)"),
                "verified": st.column_config.CheckboxColumn("Verified"),
                "profile_url": st.column_config.LinkColumn("Polymarket", display_text="Mở Profile"),
            },
        )

        dl1, dl2 = st.columns(2)
        prefix = f"top{len(df_result)}_{meta_result['category'].lower()}_{meta_result['time_period'].lower()}"

        dl1.download_button(
            "📥 Tải danh sách Top ví (CSV)",
            data=df_result.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{prefix}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl2.download_button(
            "📥 Tải Raw JSON gốc từ API",
            data=json.dumps(meta_result.get("raw_records", []), ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{prefix}_raw.json",
            mime="application/json",
            use_container_width=True,
        )

    if meta_result.get("raw_records"):
        with st.expander("🔍 Kiểm tra Raw JSON Schema (3 bản ghi đầu tiên từ API)", expanded=df_result.empty):
            st.json(meta_result.get("raw_records", [])[:3])
