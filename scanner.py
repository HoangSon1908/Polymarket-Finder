from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import streamlit as st

st.set_page_config(
    page_title="Polymarket Pro Wallet Finder",
    page_icon="🎯",
    layout="wide",
)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
TRADES_V2_URL = "https://data-api.polymarket.com/v2/trades"
LEADERBOARD_V1_URL = "https://data-api.polymarket.com/v1/leaderboard"
POSITIONS_V2_URL = "https://data-api.polymarket.com/v2/positions"
ACTIVITY_V2_URL = "https://data-api.polymarket.com/v2/activity"
USER_STATS_V2_URL = "https://data-api.polymarket.com/v2/user-stats"
USER_VOLUME_V2_URL = "https://data-api.polymarket.com/v2/user-volume"
USER_PNL_V2_URL = "https://data-api.polymarket.com/v2/user-pnl"

DEFAULT_BENCHMARK_WALLET = "0xc9bfc7986767173b4f1c3fc9b2c6b255fc7381c0"
VN_TZ = timezone(timedelta(hours=7))

_THREAD_LOCAL = threading.local()


# ==============================================================================
# 1. HTTP SESSION POOLING & DATA API V2 HELPERS
# ==============================================================================

def get_http_session() -> requests.Session:
    """Mỗi worker thread giữ 1 requests.Session Keep-Alive để tái sử dụng kết nối TLS."""
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _THREAD_LOCAL.session = sess
    return sess


def request_with_retry(url: str, params: dict, retries: int = 3, timeout: int = 12):
    sess = get_http_session()
    last_err = None
    for attempt in range(retries):
        try:
            res = sess.get(url, params=params, timeout=timeout)
            if res.status_code == 429:
                time.sleep(1.2 * (attempt + 1))
                continue
            res.raise_for_status()
            return res.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.45 * (attempt + 1))
    raise RuntimeError(f"Lỗi gọi {url}: {last_err}")


def fetch_v2_paginated(url: str, wallet: str, extra_params: dict | None = None, max_pages: int = 12) -> list[dict]:
    """Kéo dữ liệu chuẩn Polymarket Data API v2 bằng phân trang cursor (không dùng offset)."""
    rows: list[dict] = []
    base_params = {"user": wallet.strip().lower(), "limit": 500}
    if extra_params:
        base_params.update(extra_params)

    params = dict(base_params)
    for _ in range(max_pages):
        payload = request_with_retry(url, params=params)
        if not payload:
            break

        if isinstance(payload, list):
            rows.extend(payload)
            break

        batch = payload.get("data") or []
        rows.extend(batch)

        pagination = payload.get("pagination") or {}
        next_cursor = pagination.get("next_cursor")
        has_more = pagination.get("has_more", False)

        if not next_cursor or not has_more:
            break

        params = dict(base_params)
        params.pop("limit", None)
        params["cursor"] = next_cursor

    return rows


# ==============================================================================
# 2. TẦNG 0: THU THẬP VÍ HẠT GIỐNG TỰ NHIÊN (MULTI-CLIP SEEDING)
# ==============================================================================

def seed_wallets_from_markets(
    category_filter: str, num_markets: int, min_accum_per_market: float
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Đào ví tự nhiên từ Gamma API (/events) + Data API v2 (/v2/trades) theo gom lệnh đa clip."""
    logs: list[str] = []
    raw_markets: list[dict] = []

    for offset in (0, 100):
        try:
            ev_list = request_with_retry(
                GAMMA_EVENTS_URL,
                {
                    "limit": 100,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                    "active": "true",
                    "closed": "false",
                },
            ) or []
            for ev in ev_list:
                ev_title = str(ev.get("title") or "")
                ev_slug = str(ev.get("slug") or "")
                ev_cat = str(ev.get("category") or "")
                for m in ev.get("markets") or []:
                    m_copy = dict(m)
                    m_copy["_ev_title"] = ev_title
                    m_copy["_ev_slug"] = ev_slug
                    m_copy["_ev_cat"] = ev_cat
                    raw_markets.append(m_copy)
            if len(ev_list) < 100:
                break
        except Exception as e:
            logs.append(f"Cảnh báo Gamma /events (offset={offset}): {e}")
            break

    kw_map = {
        "Esports (CS2 / Dota / LoL / Valorant)": [
            "counter-strike", "cs2", "dota", "league of legends", "lol:", "valorant",
            "esports", "map 1", "map 2", "map handicap", "blast", "iem", "pgl", "cct"
        ],
        "Sports Truyền Thống (NBA / Tennis / Bóng đá / UFC)": [
            "nba", "nfl", "nhl", "mlb", "ncaa", "atp", "wta", "tennis", "premier league",
            "champions league", "la liga", "serie a", "bundesliga", "ligue 1", "soccer",
            "football", "ufc", "mma", "f1", "formula 1", "set 1", "set handicap"
        ],
        "Tất cả Sports & Esports": [
            "nba", "nfl", "nhl", "mlb", "ncaa", "atp", "wta", "tennis", "premier league",
            "champions league", "la liga", "serie a", "bundesliga", "soccer", "football",
            "counter-strike", "cs2", "dota", "league of legends", "valorant", "ufc", "f1",
            "vs.", "vs ", "spread:", "o/u "
        ],
        "Finance (Chỉ số / Cổ phiếu / Vàng / Lãi suất)": [
            "s&p", "spx", "nasdaq", "dow jones", "nvda", "tsla", "aapl", "msft", "amzn",
            "gold", "silver", "oil", "crude", "fed", "fomc", "interest rate", "cpi",
            "inflation", "treasury", "close above", "close below", "stock", "earnings"
        ],
        "Politics & Macro": [
            "trump", "biden", "harris", "election", "senate", "house", "president",
            "tariff", "minister", "parliament", "vote", "supreme court", "governor"
        ],
    }

    seen_cids: set[str] = set()
    filtered_markets: list[dict] = []
    for m in raw_markets:
        if m.get("closed") is True:
            continue
        cid = str(m.get("conditionId") or m.get("condition_id") or "").strip()
        if not cid or cid in seen_cids:
            continue

        full_text = f"{m.get('question', '')} {m.get('_ev_title', '')} {m.get('_ev_slug', '')} {m.get('_ev_cat', '')}".lower()
        if any(k in full_text for k in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "up or down"]):
            continue

        kws = kw_map.get(category_filter, [])
        if any(k in full_text for k in kws):
            seen_cids.add(cid)
            filtered_markets.append(m)

    logs.append(f"Gamma API: Tìm thấy **{len(filtered_markets)}** market khớp nhóm `{category_filter}`.")
    target_markets = filtered_markets[:num_markets]

    pair_usdc: dict[tuple[str, str], float] = defaultdict(float)
    pair_clips: dict[tuple[str, str], int] = defaultdict(int)
    cid_title_map: dict[str, str] = {}

    def fetch_market_trades(m: dict) -> tuple[str, str, list[dict]]:
        cid = str(m.get("conditionId") or m.get("condition_id") or "").strip()
        title = str(m.get("question") or m.get("_ev_title") or "")
        for param_key in ("condition", "market"):
            try:
                payload = request_with_retry(TRADES_V2_URL, params={param_key: cid, "limit": 500})
                batch = payload.get("data") or [] if isinstance(payload, dict) else (payload or [])
                if batch:
                    return cid, title, batch
            except Exception:
                continue
        return cid, title, []

    # Kéo /v2/trades song song trên 6 luồng
    with ThreadPoolExecutor(max_workers=6) as pool:
        for cid, title, trades_batch in pool.map(fetch_market_trades, target_markets):
            cid_title_map[cid] = title
            for t in trades_batch:
                if not isinstance(t, dict):
                    continue
                w = str(
                    t.get("proxy_wallet")
                    or t.get("proxyWallet")
                    or t.get("user")
                    or t.get("taker_address")
                    or ""
                ).strip().lower()
                if not w.startswith("0x") or len(w) != 42:
                    continue
                size = float(t.get("size") or t.get("shares") or 0)
                price = float(t.get("price") or 0)
                usdc = float(t.get("usdc_size") or t.get("usdcSize") or (size * price))
                if usdc > 0:
                    pair_usdc[(w, cid)] += usdc
                    pair_clips[(w, cid)] += 1

    wallet_agg: dict[str, dict] = {}
    for (w, cid), tot_usdc in pair_usdc.items():
        if tot_usdc < min_accum_per_market:
            continue
        rec = wallet_agg.setdefault(
            w,
            {
                "wallet": w,
                "qualifying_markets": 0,
                "total_clips": 0,
                "seed_usdc": 0.0,
                "max_single_market_usdc": 0.0,
                "sample_market": cid_title_map.get(cid, ""),
            },
        )
        rec["qualifying_markets"] += 1
        rec["total_clips"] += pair_clips[(w, cid)]
        rec["seed_usdc"] += tot_usdc
        if tot_usdc > rec["max_single_market_usdc"]:
            rec["max_single_market_usdc"] = tot_usdc
            rec["sample_market"] = cid_title_map.get(cid, "")

    seed_df = pd.DataFrame(list(wallet_agg.values()))
    if not seed_df.empty:
        seed_df = seed_df.sort_values(["seed_usdc", "total_clips"], ascending=[False, False]).reset_index(drop=True)
        logs.append(f"Data API `/v2/trades`: Thu thập tự nhiên được **{len(seed_df)}** ví có tổng gom >= ${min_accum_per_market:,.0f}/trận.")
        return seed_df["wallet"].tolist(), seed_df, logs

    logs.append("Chưa có ví nào đạt ngưỡng gom tối thiểu trên lô trận vừa quét.")
    return [], pd.DataFrame(), logs


def seed_wallets_from_leaderboard(category: str, time_period: str, limit: int) -> tuple[list[str], pd.DataFrame, list[str]]:
    logs: list[str] = []
    rows = []
    fetched = 0
    offset = 0

    while fetched < limit:
        batch_size = min(50, limit - fetched)
        params = {
            "category": category,
            "timePeriod": time_period,
            "orderBy": "PNL",
            "limit": batch_size,
            "offset": offset,
        }
        try:
            payload = request_with_retry(LEADERBOARD_V1_URL, params=params)
        except Exception as e:
            logs.append(f"Lỗi Leaderboard (offset={offset}): {e}")
            break

        data = payload.get("data") or payload.get("results") or [] if isinstance(payload, dict) else (payload or [])
        if not data:
            break

        for idx_in_batch, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            w = str(
                item.get("proxy_wallet")
                or item.get("proxyWallet")
                or item.get("user")
                or item.get("address")
                or ""
            ).strip().lower()
            if w.startswith("0x") and len(w) == 42:
                rows.append({
                    "lb_rank": offset + idx_in_batch + 1,
                    "wallet": w,
                    "userName": item.get("user_name") or item.get("userName") or item.get("name") or "",
                    "lb_pnl": float(item.get("pnl") or 0),
                    "lb_vol": float(item.get("vol") or item.get("volume") or 0),
                })

        fetched += len(data)
        offset += batch_size
        if len(data) < batch_size:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["wallet"]).reset_index(drop=True) if rows else pd.DataFrame()
    if not df.empty:
        logs.append(f"Leaderboard ({category} / {time_period}): Kéo tự nhiên thành công **{len(df)}** ví.")
    return (df["wallet"].tolist() if not df.empty else []), df, logs


# ==============================================================================
# 3. KIỂM TOÁN 3 CHẶN TIẾN TIẾN (FAST SURFACE -> POSITIONS -> DEEP ACTIVITY)
# ==============================================================================

def fast_surface_check(wallet: str, cfg: dict) -> tuple[bool, str, dict]:
    """
    Chặng 1 (~0.25s): Gọi các endpoint tổng hợp v2 (/user-stats, /user-volume, /user-pnl)
    để loại ngay ví mới, bot MM quá lớn, hoặc ví âm/lãi quá mỏng trước khi kéo sổ chi tiết.
    """
    w = wallet.strip().lower()
    stats_data = {}
    vol_data = {}
    pnl_data = {}

    try:
        r1 = request_with_retry(USER_STATS_V2_URL, {"user": w}, retries=2, timeout=8)
        if isinstance(r1, dict):
            stats_data = r1.get("data") if isinstance(r1.get("data"), dict) else r1
    except Exception:
        pass

    try:
        r2 = request_with_retry(USER_VOLUME_V2_URL, {"user": w}, retries=2, timeout=8)
        if isinstance(r2, dict):
            vol_data = r2.get("data") if isinstance(r2.get("data"), dict) else r2
    except Exception:
        pass

    try:
        r3 = request_with_retry(USER_PNL_V2_URL, {"user": w}, retries=2, timeout=8)
        if isinstance(r3, dict):
            pnl_data = r3.get("data") if isinstance(r3.get("data"), dict) else r3
    except Exception:
        pass

    def pick_num(d: dict, keys: list[str]) -> float | None:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return None

    m_traded = pick_num(stats_data, ["markets_traded", "marketsTraded", "total_markets", "markets"])
    ui_vol = pick_num(vol_data, ["volume", "total_volume", "totalVolume", "usdc_volume"])
    ui_pnl = pick_num(pnl_data, ["pnl", "total_pnl", "totalPnl", "net_pnl"])
    if ui_pnl is None:
        ui_pnl = pick_num(stats_data, ["pnl", "total_pnl", "totalPnl"])

    meta = {"ui_markets": m_traded, "ui_vol": ui_vol, "ui_pnl": ui_pnl}

    # Chỉ chặn sớm khi API trả về số liệu rõ ràng vi phạm ngưỡng (có vùng đệm an toàn 20%)
    if m_traded is not None and m_traded > 0:
        if m_traded < cfg["min_markets"] * 0.8:
            return False, f"⚡ Chặn sớm Tầng 1: Ít mẫu ({int(m_traded)} < {cfg['min_markets']} kèo)", meta
        if m_traded > cfg["max_markets"] * 1.2:
            return False, f"⚡ Chặn sớm Tầng 1: Đánh quá rộng ({int(m_traded)} > {cfg['max_markets']} kèo, nghi bot MM)", meta

    if ui_pnl is not None and ui_pnl < cfg["min_net_pnl"] * 0.7:
        return False, f"⚡ Chặn sớm Tầng 1: Net PnL thấp (${ui_pnl:,.0f} < ${cfg['min_net_pnl']:,.0f})", meta

    if ui_pnl is not None and ui_vol is not None and ui_vol > 1000:
        surf_roi = (ui_pnl / ui_vol) * 100.0
        # UI Volume tính 2 chiều nên surf_roi ~ 0.5 * True ROI; chặn nếu thấp hơn 35% mục tiêu
        if surf_roi < cfg["min_true_roi"] * 0.35:
            return False, f"⚡ Chặn sớm Tầng 1: PnL/Volume bề mặt quá mỏng ({surf_roi:+.2f}%)", meta

    return True, "", meta


def fetch_positions_only_df(wallet: str, max_pages: int = 12) -> pd.DataFrame:
    """Chặng 2A: Kéo 4 trạng thái từ /v2/positions và khử trùng lặp."""
    statuses = ["OPEN", "CLOSED", "REDEEMABLE", "REDEEMABLE_LOST"]
    raw_pos: list[dict] = []
    for st_val in statuses:
        try:
            batch = fetch_v2_paginated(POSITIONS_V2_URL, wallet, {"status": st_val}, max_pages=max_pages)
            for item in batch:
                item["_queried_status"] = st_val
            raw_pos.extend(batch)
        except Exception:
            continue

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
    pos_df["status"] = get_col(pos_df, ["status", "_queried_status"], "CLOSED").astype(str).str.upper()

    pos_df["avg_price"] = pd.to_numeric(get_col(pos_df, ["avg_price", "avgPrice"]), errors="coerce").fillna(0.0)
    pos_df["entry_cost_usdc"] = pd.to_numeric(get_col(pos_df, ["entry_cost_usdc", "initialValue"]), errors="coerce").fillna(0.0)
    pos_df["current_size"] = pd.to_numeric(get_col(pos_df, ["current_size", "size"]), errors="coerce").fillna(0.0)
    pos_df["current_value"] = pd.to_numeric(get_col(pos_df, ["current_value", "currentValue"]), errors="coerce").fillna(0.0)
    pos_df["realized_pnl"] = pd.to_numeric(get_col(pos_df, ["realized_pnl", "realizedPnl"]), errors="coerce").fillna(0.0)
    pos_df["unrealized_pnl"] = pd.to_numeric(get_col(pos_df, ["unrealized_pnl", "unrealizedPnl"]), errors="coerce").fillna(0.0)
    pos_df["cur_price"] = pd.to_numeric(get_col(pos_df, ["current_price", "curPrice"]), errors="coerce").fillna(0.0)

    slug_s = get_col(pos_df, ["event_slug", "eventSlug", "slug"], "")
    pos_df["url"] = slug_s.apply(lambda s: f"https://polymarket.com/event/{s}" if pd.notna(s) and str(s).strip() else "")

    status_rank = {"OPEN": 1, "REDEEMABLE": 2, "REDEEMABLE_LOST": 2, "CLOSED": 3}
    pos_df["_rank"] = pos_df["status"].map(status_rank).fillna(9)
    pos_df = (
        pos_df.sort_values(["_rank", "entry_cost_usdc"], ascending=[True, False])
        .drop_duplicates(subset=["token_id", "outcome"], keep="first")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    pos_df["total_pnl"] = pos_df["realized_pnl"] + pos_df["unrealized_pnl"]
    pos_df["real_win"] = pos_df["total_pnl"] > 0
    return pos_df


def enrich_with_activity_df(wallet: str, pos_df: pd.DataFrame, max_pages: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chặng 2B: Kéo /v2/activity và ghép vốn mua thực 1 chiều + hành vi gom/thoát lệnh."""
    def get_col(df: pd.DataFrame, names: list[str], default=None) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df), index=df.index)

    raw_act: list[dict] = []
    try:
        raw_act = fetch_v2_paginated(ACTIVITY_V2_URL, wallet, {}, max_pages=max_pages)
    except Exception:
        raw_act = []

    act_df = pd.DataFrame(raw_act) if raw_act else pd.DataFrame()
    act_agg_map: dict[str, dict] = {}

    if not act_df.empty:
        act_df["token_id"] = get_col(act_df, ["token_id", "asset", "asset_id"], "").astype(str).str.strip()
        act_df["type"] = get_col(act_df, ["type", "activity_type"], "TRADE").astype(str).str.upper()
        act_df["side"] = get_col(act_df, ["side"], "").astype(str).str.upper()
        act_df["price"] = pd.to_numeric(get_col(act_df, ["price"]), errors="coerce").fillna(0.0)
        act_df["size"] = pd.to_numeric(get_col(act_df, ["size", "shares"]), errors="coerce").fillna(0.0)
        act_df["usdc"] = pd.to_numeric(get_col(act_df, ["usdc_size", "usdcSize", "amount"]), errors="coerce").fillna(0.0)
        act_df["usdc"] = np.where(act_df["usdc"] > 0, act_df["usdc"], act_df["price"] * act_df["size"])
        act_df["ts"] = pd.to_numeric(get_col(act_df, ["timestamp", "created_at", "time"]), errors="coerce")

        dedup_cols = [c for c in ["id", "transaction_hash", "transactionHash", "token_id", "ts", "side", "usdc"] if c in act_df.columns]
        if dedup_cols:
            act_df = act_df.drop_duplicates(subset=dedup_cols)

        for tid, grp in act_df.groupby("token_id"):
            if not tid:
                continue
            buys = grp[(grp["type"].str.contains("TRADE|BUY")) & (grp["side"] == "BUY")]
            sells = grp[(grp["type"].str.contains("TRADE|SELL")) & (grp["side"] == "SELL")]
            redeems = grp[grp["type"].str.contains("REDEEM")]

            buy_usdc = float(buys["usdc"].sum())
            sell_usdc = float(sells["usdc"].sum())
            redeem_usdc = float(redeems["usdc"].sum())
            buy_cnt = int(len(buys))
            sell_cnt = int(len(sells))

            first_buy_ts = buys["ts"].min() if not buys.empty else grp["ts"].min()
            last_buy_ts = buys["ts"].max() if not buys.empty else grp["ts"].max()
            last_ev_ts = grp["ts"].max()

            accum_h = max(0.0, (last_buy_ts - first_buy_ts) / 3600.0) if pd.notna(first_buy_ts) and pd.notna(last_buy_ts) else 0.0
            hold_h = max(0.0, (last_ev_ts - first_buy_ts) / 3600.0) if pd.notna(first_buy_ts) and pd.notna(last_ev_ts) else 0.0

            act_agg_map[tid] = {
                "act_buy_usdc": buy_usdc,
                "act_sell_usdc": sell_usdc,
                "act_redeem_usdc": redeem_usdc,
                "buy_count": buy_cnt,
                "sell_count": sell_cnt,
                "accum_hours": accum_h,
                "hold_hours": hold_h,
                "first_ts": first_buy_ts,
                "last_ts": last_ev_ts,
            }

    out_df = pos_df.copy()
    act_records = [act_agg_map.get(tid, {}) for tid in out_df["token_id"]]
    act_merged = pd.DataFrame(act_records, index=out_df.index)

    for col, default_v in [
        ("act_buy_usdc", 0.0),
        ("act_sell_usdc", 0.0),
        ("act_redeem_usdc", 0.0),
        ("buy_count", 1),
        ("sell_count", 0),
        ("accum_hours", 0.0),
        ("hold_hours", 0.0),
        ("first_ts", np.nan),
        ("last_ts", np.nan),
    ]:
        if col in act_merged.columns:
            out_df[col] = act_merged[col].fillna(default_v)
        else:
            out_df[col] = default_v

    out_df["true_cost"] = np.where(out_df["act_buy_usdc"] > 0, out_df["act_buy_usdc"], out_df["entry_cost_usdc"])
    out_df["roi_pct"] = np.where(out_df["true_cost"] > 0, out_df["total_pnl"] / out_df["true_cost"] * 100.0, 0.0)

    cid_counts = out_df[out_df["condition_id"] != ""].groupby("condition_id")["outcome"].nunique()
    hedged_cids = set(cid_counts[cid_counts >= 2].index)
    out_df["is_hedged"] = out_df["condition_id"].isin(hedged_cids)

    def classify_exit(r) -> str:
        if r["status"] == "OPEN":
            return "OPEN"
        if r["sell_count"] == 0:
            return "HOLD_TO_EXPIRY"
        if r["act_redeem_usdc"] == 0 and (0.01 < r["cur_price"] < 0.99 or r["act_sell_usdc"] > 0):
            return "SOLD_EARLY"
        if r["act_redeem_usdc"] > 0 and r["act_sell_usdc"] < 0.25 * max(r["true_cost"], 1e-6):
            return "HOLD_TO_EXPIRY"
        return "PARTIAL_SELL"

    out_df["exit_style"] = out_df.apply(classify_exit, axis=1)
    out_df["entry_time_vn"] = pd.to_datetime(out_df["first_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    out_df["last_time_vn"] = pd.to_datetime(out_df["last_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    out_df["_activity_enriched"] = True

    return out_df, act_df


def make_empty_summary(wallet: str, is_bench: bool, reason: str, markets: int = 0, net_pnl: float = 0.0, true_cost: float = 0.0) -> dict:
    prefix = "🎯 [VÍ MẪU] " if is_bench else ""
    return {
        "wallet": wallet,
        "is_benchmark": is_bench,
        "passed_filter": False,
        "verdict": f"{prefix}❌ LOẠI",
        "score": 0,
        "markets": markets,
        "positions": 0,
        "true_cost": round(true_cost, 2),
        "net_pnl": round(net_pnl, 2),
        "true_roi": round((net_pnl / true_cost * 100.0) if true_cost > 0 else 0.0, 2),
        "count_wr": 0.0,
        "cap_wr": 0.0,
        "cap_edge": 0.0,
        "z_score": 0.0,
        "hold_pnl_share": 0.0,
        "core_band_share": 0.0,
        "penny_share": 0.0,
        "high_fav_share": 0.0,
        "pnl_ex_top5": 0.0,
        "top20_roi": 0.0,
        "med_size": 0.0,
        "med_accum_h": 0.0,
        "med_hold_h": 0.0,
        "hedged_pct": 0.0,
        "red_flags": reason,
    }


def evaluate_wallet_pro(
    wallet: str,
    cfg: dict,
    max_pages: int = 12,
    is_benchmark_wallet: bool = False,
    fast_short_circuit: bool = True,
    cached_full_df: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Pipeline kiểm toán 3 chặng có ngắt mạch sớm (Short-Circuit):
    1) Fast Surface Check (/user-stats, /user-volume, /user-pnl)
    2) /v2/positions Check (chặn sớm nếu lệch số kèo, PnL, hoặc dải giá trước khi kéo /v2/activity)
    3) Deep /v2/activity Audit (tính chính xác vốn 1 chiều, % REDEEM, Z-score, PnL ex-Top 5)
    """
    if cached_full_df is not None and not cached_full_df.empty and cached_full_df.get("_activity_enriched", pd.Series([False])).iloc[0]:
        pos_df = cached_full_df
    else:
        # Chặng 1: Kiểm tra bề mặt siêu tốc
        if fast_short_circuit and not is_benchmark_wallet:
            ok_surf, reason_surf, meta_surf = fast_surface_check(wallet, cfg)
            if not ok_surf:
                return make_empty_summary(
                    wallet,
                    is_benchmark_wallet,
                    reason_surf,
                    markets=int(meta_surf["ui_markets"] or 0),
                    net_pnl=float(meta_surf["ui_pnl"] or 0.0),
                ), pd.DataFrame()

        # Chặng 2A: Kéo /v2/positions
        raw_pos_df = fetch_positions_only_df(wallet, max_pages=max_pages)
        if raw_pos_df.empty:
            return make_empty_summary(wallet, is_benchmark_wallet, "Ví không có lịch sử vị thế"), pd.DataFrame()

        # Kiểm tra nhanh các điều kiện cứng trên /v2/positions trước khi gọi /v2/activity nặng
        if fast_short_circuit and not is_benchmark_wallet:
            n_mkt_pre = int(raw_pos_df["condition_id"].replace("", np.nan).nunique()) or len(raw_pos_df)
            pnl_pre = float(raw_pos_df["total_pnl"].sum())
            cost_pre = float(raw_pos_df["entry_cost_usdc"].sum())

            if n_mkt_pre < cfg["min_markets"]:
                return make_empty_summary(
                    wallet, is_benchmark_wallet,
                    f"⚡ Dừng sau /positions: Ít mẫu ({n_mkt_pre} < {cfg['min_markets']} kèo)",
                    markets=n_mkt_pre, net_pnl=pnl_pre, true_cost=cost_pre,
                ), pd.DataFrame()
            if n_mkt_pre > cfg["max_markets"]:
                return make_empty_summary(
                    wallet, is_benchmark_wallet,
                    f"⚡ Dừng sau /positions: Đánh quá rộng ({n_mkt_pre} > {cfg['max_markets']} kèo)",
                    markets=n_mkt_pre, net_pnl=pnl_pre, true_cost=cost_pre,
                ), pd.DataFrame()
            if pnl_pre < cfg["min_net_pnl"]:
                return make_empty_summary(
                    wallet, is_benchmark_wallet,
                    f"⚡ Dừng sau /positions: Net PnL thấp (${pnl_pre:,.0f} < ${cfg['min_net_pnl']:,.0f})",
                    markets=n_mkt_pre, net_pnl=pnl_pre, true_cost=cost_pre,
                ), pd.DataFrame()

            # Kiểm tra phân bổ dải giá theo số lượng / entry_cost trước khi kéo activity
            price_weights = np.where(raw_pos_df["entry_cost_usdc"] > 0, raw_pos_df["entry_cost_usdc"], 1.0)
            tot_w = float(np.sum(price_weights))
            fav_w = float(np.sum(price_weights[raw_pos_df["avg_price"] >= 0.95])) / tot_w * 100.0
            if fav_w > cfg["max_high_fav_share"] * 1.3:
                return make_empty_summary(
                    wallet, is_benchmark_wallet,
                    f"⚡ Dừng sau /positions: Nghi bẫy Maker/Fav >=95¢ ({fav_w:.0f}%)",
                    markets=n_mkt_pre, net_pnl=pnl_pre, true_cost=cost_pre,
                ), pd.DataFrame()

        # Chặng 2B: Đã qua mọi chốt chặn -> Kéo /v2/activity đầy đủ
        pos_df, _ = enrich_with_activity_df(wallet, raw_pos_df, max_pages=max_pages)

    n_markets = int(pos_df["condition_id"].replace("", np.nan).nunique())
    if n_markets == 0:
        n_markets = len(pos_df)
    n_pos = len(pos_df)

    total_cost = float(pos_df["true_cost"].sum())
    net_pnl = float(pos_df["total_pnl"].sum())
    true_roi = (net_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

    count_wr = float(pos_df["real_win"].mean() * 100.0)
    cap_wr = float(pos_df.loc[pos_df["real_win"], "true_cost"].sum() / total_cost * 100.0) if total_cost > 0 else 0.0
    w_entry = float((pos_df["avg_price"] * pos_df["true_cost"]).sum() / total_cost * 100.0) if total_cost > 0 else 0.0
    cap_edge = cap_wr - w_entry

    if total_cost > 0 and n_pos >= 10:
        weights = pos_df["true_cost"] / total_cost
        probs = pos_df["avg_price"].clip(0.005, 0.995)
        var_cap = float(np.sum((weights ** 2) * probs * (1.0 - probs)))
        z_score = ((cap_wr - w_entry) / 100.0) / math.sqrt(var_cap) if var_cap > 0 else 0.0
    else:
        z_score = 0.0

    hold_df = pos_df[pos_df["exit_style"] == "HOLD_TO_EXPIRY"]
    early_df = pos_df[pos_df["exit_style"] == "SOLD_EARLY"]
    open_df = pos_df[pos_df["exit_style"] == "OPEN"]

    hold_pnl = float(hold_df["total_pnl"].sum())
    early_pnl = float(early_df["total_pnl"].sum())
    open_unrealized = float(open_df["unrealized_pnl"].sum())

    if net_pnl > 0:
        hold_pnl_share = max(0.0, min(150.0, hold_pnl / net_pnl * 100.0))
    else:
        hold_pnl_share = 0.0

    core_band_cost = float(pos_df[(pos_df["avg_price"] >= 0.20) & (pos_df["avg_price"] <= 0.85)]["true_cost"].sum())
    penny_cost = float(pos_df[pos_df["avg_price"] < 0.10]["true_cost"].sum())
    high_fav_cost = float(pos_df[pos_df["avg_price"] >= 0.95]["true_cost"].sum())

    core_band_share = (core_band_cost / total_cost * 100.0) if total_cost > 0 else 0.0
    penny_share = (penny_cost / total_cost * 100.0) if total_cost > 0 else 0.0
    high_fav_share = (high_fav_cost / total_cost * 100.0) if total_cost > 0 else 0.0

    top5_wins = float(pos_df[pos_df["total_pnl"] > 0].sort_values("total_pnl", ascending=False)["total_pnl"].head(5).sum())
    pnl_ex_top5 = net_pnl - top5_wins

    q80 = pos_df["true_cost"].quantile(0.80)
    q50 = pos_df["true_cost"].quantile(0.50)
    top20_df = pos_df[pos_df["true_cost"] >= q80]
    bot50_df = pos_df[pos_df["true_cost"] <= q50]
    top20_roi = (top20_df["total_pnl"].sum() / top20_df["true_cost"].sum() * 100.0) if top20_df["true_cost"].sum() > 0 else 0.0
    bot50_roi = (bot50_df["total_pnl"].sum() / bot50_df["true_cost"].sum() * 100.0) if bot50_df["true_cost"].sum() > 0 else 0.0

    med_size = float(pos_df["true_cost"].median())
    med_accum_h = float(pos_df["accum_hours"].median())
    med_hold_h = float(pos_df["hold_hours"].median())
    hedged_pct = float(pos_df["is_hedged"].mean() * 100.0)

    flags = []
    score = 0

    if cfg["min_markets"] <= n_markets <= cfg["max_markets"]:
        score += 10
    elif n_markets < cfg["min_markets"]:
        flags.append(f"Ít mẫu ({n_markets} < {cfg['min_markets']} kèo)")
    else:
        flags.append(f"Đánh quá rộng ({n_markets} > {cfg['max_markets']} kèo, nghi bot MM)")

    if net_pnl >= cfg["min_net_pnl"] and true_roi >= cfg["min_true_roi"]:
        score += 20
    else:
        if net_pnl < cfg["min_net_pnl"]:
            flags.append(f"Lãi ròng thấp (${net_pnl:,.0f} < ${cfg['min_net_pnl']:,.0f})")
        if true_roi < cfg["min_true_roi"]:
            flags.append(f"True ROI thấp ({true_roi:+.1f}% < {cfg['min_true_roi']:.1f}%)")

    if open_unrealized < -0.25 * max(abs(net_pnl), 1000):
        flags.append(f"Đang gồng lỗ OPEN (${open_unrealized:+,.0f})")

    if cap_edge >= cfg["min_cap_edge"] and z_score >= cfg["min_z_score"]:
        score += 20
    elif cap_edge >= cfg["min_cap_edge"] * 0.7 and z_score >= 1.28:
        score += 10
        flags.append(f"Edge/Z-score dưới ngưỡng (Edge {cap_edge:+.1f}%, Z={z_score:.2f})")
    else:
        flags.append(f"Thiếu Edge thống kê (CapEdge {cap_edge:+.1f}%, Z={z_score:.2f})")

    if hold_pnl_share >= cfg["min_hold_pnl_share"]:
        score += 20
    else:
        flags.append(f"Lãi phụ thuộc bán non SOLD_EARLY (Hold chỉ chiếm {hold_pnl_share:.0f}%, Early=${early_pnl:+,.0f})")

    if core_band_share >= cfg["min_core_band_share"] and high_fav_share <= cfg["max_high_fav_share"] and penny_share <= 20.0:
        score += 15
    else:
        if high_fav_share > cfg["max_high_fav_share"]:
            flags.append(f"Bẫy Maker/Fav >=95¢ ({high_fav_share:.1f}% vốn)")
        if penny_share > 20.0:
            flags.append(f"Đánh vé số <10¢ quá nhiều ({penny_share:.1f}% vốn)")
        if core_band_share < cfg["min_core_band_share"]:
            flags.append(f"Vốn dải chuẩn 20-85% thấp ({core_band_share:.1f}%)")

    if pnl_ex_top5 > 0:
        score += 8
    else:
        flags.append(f"Phụ thuộc Top 5 kèo độc đắc (PnL ex-Top 5 = ${pnl_ex_top5:+,.0f})")

    if cap_wr >= count_wr and top20_roi >= bot50_roi:
        score += 7
    else:
        flags.append(f"Nhồi tiền kém hiệu quả (CapWR {cap_wr:.1f}% vs WR {count_wr:.1f}%, Top20% ROI {top20_roi:+.1f}%)")

    if hedged_pct > 20.0:
        flags.append(f"Hedge 2 đầu cao ({hedged_pct:.1f}% trận)")

    critical_fail = (
        n_markets < cfg["min_markets"]
        or n_markets > cfg["max_markets"]
        or net_pnl < cfg["min_net_pnl"]
        or true_roi < cfg["min_true_roi"]
        or cap_edge < cfg["min_cap_edge"]
        or z_score < cfg["min_z_score"]
        or hold_pnl_share < cfg["min_hold_pnl_share"]
        or core_band_share < cfg["min_core_band_share"]
        or high_fav_share > cfg["max_high_fav_share"]
        or pnl_ex_top5 <= 0
    )

    passed_filter = (score >= 80) and (not critical_fail)
    prefix = "🎯 [VÍ MẪU] " if is_benchmark_wallet else ""
    if passed_filter:
        verdict = f"{prefix}✅ ĐẠT CHUẨN"
    elif score >= 60:
        verdict = f"{prefix}🟡 CẬN CHUẨN"
    else:
        verdict = f"{prefix}❌ LOẠI"

    summary = {
        "wallet": wallet,
        "is_benchmark": is_benchmark_wallet,
        "passed_filter": passed_filter,
        "verdict": verdict,
        "score": score,
        "markets": n_markets,
        "positions": n_pos,
        "true_cost": round(total_cost, 2),
        "net_pnl": round(net_pnl, 2),
        "true_roi": round(true_roi, 2),
        "count_wr": round(count_wr, 2),
        "cap_wr": round(cap_wr, 2),
        "cap_edge": round(cap_edge, 2),
        "z_score": round(z_score, 2),
        "hold_pnl_share": round(hold_pnl_share, 1),
        "core_band_share": round(core_band_share, 1),
        "penny_share": round(penny_share, 1),
        "high_fav_share": round(high_fav_share, 1),
        "pnl_ex_top5": round(pnl_ex_top5, 2),
        "top20_roi": round(top20_roi, 2),
        "med_size": round(med_size, 2),
        "med_accum_h": round(med_accum_h, 2),
        "med_hold_h": round(med_hold_h, 2),
        "hedged_pct": round(hedged_pct, 1),
        "red_flags": " | ".join(flags) if flags else "Sạch lỗi — Vượt qua 100% quy tắc sàng lọc",
    }
    return summary, pos_df


# ==============================================================================
# 4. STREAMLIT UI
# ==============================================================================

st.title("🎯 Trạm Săn & Kiểm Toán Ví Pro Polymarket (Tăng Tốc Đa Luồng)")
st.caption(
    "Tích hợp **Kiểm toán 3 chặng ngắt mạch sớm** (`User Stats -> Positions -> Activity`) + **6 luồng song song** "
    "và **Cache phiên làm việc**, rút ngắn thời gian quét 26–50 ví từ vài phút xuống còn vài giây."
)

with st.sidebar:
    st.header("🎯 1. Ví Mẫu Kiểm Định (Canary)")
    ref_wallet_input = st.text_input(
        "Địa chỉ Ví Mẫu dùng kiểm tra độ phủ Seeding",
        value=DEFAULT_BENCHMARK_WALLET,
    ).strip().lower()

    st.divider()
    st.header("⚡ 2. Tốc độ & Độ sâu Quét")
    c_spd1, c_spd2 = st.columns(2)
    max_api_pages = c_spd1.number_input("Số trang API (×500)", min_value=3, max_value=40, value=12, step=3)
    num_workers = c_spd2.number_input("Số luồng song song", min_value=2, max_value=10, value=6, step=1)
    fast_mode = st.checkbox(
        "Chặn sớm 3 chặng (Bỏ qua /activity với ví rớt ngưỡng)",
        value=True,
        help="Tăng tốc gấp 10x bằng cách không tải lịch sử activity nặng cho các ví đã rớt số kèo, PnL hoặc dải giá.",
    )

    st.divider()
    st.header("⚙️ 3. Ngưỡng Lọc Sổ Cái (Tầng 1 & 2)")
    c1, c2 = st.columns(2)
    min_markets = c1.number_input("Số kèo tối thiểu", min_value=10, max_value=1000, value=80, step=10)
    max_markets = c2.number_input("Số kèo tối đa", min_value=100, max_value=5000, value=1500, step=100)

    min_net_pnl = st.number_input("Net PnL tối thiểu ($)", min_value=500.0, value=5000.0, step=1000.0)
    min_true_roi = st.slider("True ROI tối thiểu trên vốn thực (%)", 2.0, 30.0, 10.0, 0.5)
    min_cap_edge = st.slider("CapEdge tối thiểu (%)", 1.0, 15.0, 4.0, 0.5)
    min_z_score = st.slider("Z-score thống kê tối thiểu", 1.0, 3.5, 1.65, 0.05)
    min_hold_pnl_share = st.slider("Tỷ trọng lãi từ đáo hạn REDEEM (%)", 50.0, 100.0, 80.0, 5.0)
    min_core_band_share = st.slider("Tỷ trọng vốn dải 20%–85% (%)", 40.0, 95.0, 70.0, 5.0)
    max_high_fav_share = st.slider("Trần vốn dải >=95¢ — Chặn Maker 99¢ (%)", 0.0, 30.0, 10.0, 1.0)

    cfg = {
        "min_markets": min_markets,
        "max_markets": max_markets,
        "min_net_pnl": min_net_pnl,
        "min_true_roi": min_true_roi,
        "min_cap_edge": min_cap_edge,
        "min_z_score": min_z_score,
        "min_hold_pnl_share": min_hold_pnl_share,
        "min_core_band_share": min_core_band_share,
        "max_high_fav_share": max_high_fav_share,
    }

for key, default_val in [
    ("audit_results", pd.DataFrame()),
    ("audit_details", {}),
    ("seed_logs", []),
    ("seed_preview", pd.DataFrame()),
    ("canary_status", None),
    ("last_elapsed_sec", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

tab_scan, tab_xray = st.tabs(["🔍 1. Đào Tự Nhiên & Sàng Lọc Ví Pro", "🔬 2. Soi Sâu Từng Ví (X-Ray & Xuất CSV)"])

with tab_scan:
    seed_mode = st.radio(
        "Chọn nguồn thu thập ví hạt giống tự nhiên:",
        options=[
            "Nguồn A: Đào từ Market thanh khoản cao (Gom đa clip trên /v2/trades)",
            "Nguồn B: Kéo từ Bảng Xếp Hạng (Leaderboard Polymarket)",
            "Nguồn C: Nhập danh sách ví thủ công / Upload CSV",
        ],
        horizontal=True,
    )

    manual_candidates: list[str] = []

    if seed_mode.startswith("Nguồn A"):
        col_a1, col_a2, col_a3, col_a4 = st.columns([2.2, 1, 1.1, 1])
        cat_choice = col_a1.selectbox(
            "Nhóm thị trường mục tiêu",
            [
                "Esports (CS2 / Dota / LoL / Valorant)",
                "Sports Truyền Thống (NBA / Tennis / Bóng đá / UFC)",
                "Tất cả Sports & Esports",
                "Finance (Chỉ số / Cổ phiếu / Vàng / Lãi suất)",
                "Politics & Macro",
            ],
        )
        n_mkts = col_a2.number_input("Số trận quét", 5, 80, 30, 5)
        min_accum_trd = col_a3.number_input("Tổng gom tối thiểu / trận ($)", 50, 20000, 1000, 100)
        max_cand_a = col_a4.number_input("Top N ví kiểm toán", 5, 100, 30, 5)

    elif seed_mode.startswith("Nguồn B"):
        col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
        lb_cat = col_b1.selectbox("Category Leaderboard", ["SPORTS", "OVERALL", "FINANCE", "POLITICS"])
        lb_period = col_b2.selectbox("Khung thời gian", ["ALL", "MONTH", "WEEK"])
        lb_limit = col_b3.number_input("Độ sâu kéo Top N ví", 20, 200, 100, 20)

    else:
        raw_text = st.text_area(
            "Dán danh sách địa chỉ ví `0x...` cần kiểm tra:",
            value="",
            placeholder="0x1234...\n0x5678...",
            height=100,
        )
        uploaded_file = st.file_uploader("Hoặc tải lên file CSV chứa cột địa chỉ ví (`0x...`)", type=["csv"])

        parsed_wallets = re.findall(r"0x[a-fA-F0-9]{40}", raw_text)
        if uploaded_file is not None:
            try:
                up_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                parsed_wallets.extend(re.findall(r"0x[a-fA-F0-9]{40}", up_content))
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")

        manual_candidates = list(dict.fromkeys(w.lower() for w in parsed_wallets))

    run_scan_clicked = st.button(
        "🚀 THU THẬP HẠT GIỐNG TỰ NHIÊN & CHẠY SÀNG LỌC",
        type="primary",
        use_container_width=True,
    )

    if run_scan_clicked:
        t0 = time.perf_counter()

        if seed_mode.startswith("Nguồn A"):
            with st.spinner("Đang quét song song Gamma API & Data API /v2/trades..."):
                raw_seeded, seed_df, logs = seed_wallets_from_markets(cat_choice, int(n_mkts), float(min_accum_trd))
                cand_list = raw_seeded[: int(max_cand_a)]
                st.session_state["seed_preview"] = seed_df
                st.session_state["seed_logs"] = logs
        elif seed_mode.startswith("Nguồn B"):
            with st.spinner("Đang kéo dữ liệu Leaderboard Polymarket..."):
                raw_seeded, seed_df, logs = seed_wallets_from_leaderboard(lb_cat, lb_period, int(lb_limit))
                cand_list = raw_seeded
                st.session_state["seed_preview"] = seed_df
                st.session_state["seed_logs"] = logs
        else:
            raw_seeded = manual_candidates
            cand_list = manual_candidates
            st.session_state["seed_preview"] = pd.DataFrame()
            st.session_state["seed_logs"] = [f"Nguồn thủ công: Nhận **{len(cand_list)}** địa chỉ ví."]

        if ref_wallet_input in raw_seeded:
            rank_in_seed = raw_seeded.index(ref_wallet_input) + 1
            in_audit_slice = ref_wallet_input in cand_list
            st.session_state["canary_status"] = (
                "FOUND",
                f"🎯 **SEEDING TỰ NHIÊN ĐÃ BẮT TRÚNG VÍ MẪU:** `{ref_wallet_input}` xuất hiện tự nhiên ở **hạng #{rank_in_seed}/{len(raw_seeded)}** "
                + ("và sẽ tự đi qua các tầng kiểm toán bên dưới." if in_audit_slice else f"(nằm ngoài Top {len(cand_list)} cắt vào kiểm toán)."),
            )
        else:
            st.session_state["canary_status"] = (
                "NOT_FOUND",
                f"ℹ️ **Ví Mẫu `{ref_wallet_input[:10]}...` không có mặt trong tập hạt giống vừa thu thập tự nhiên** "
                "(do khác category mục tiêu hoặc chưa phát sinh giao dịch mới trong lô trận vừa quét).",
            )

        if not cand_list:
            st.warning("⚠️ Bộ thu thập tự nhiên trả về 0 ví. Hãy giảm ngưỡng 'Tổng gom tối thiểu / trận ($)' hoặc tăng độ sâu quét.")
        else:
            results = []
            details_map = dict(st.session_state["audit_details"])
            prog = st.progress(0.0, text=f"Đang kiểm toán song song {len(cand_list)} ví trên {int(num_workers)} luồng...")

            def worker_task(wallet_addr: str):
                is_bench = (wallet_addr == ref_wallet_input)
                cached_df = details_map.get(wallet_addr)
                return evaluate_wallet_pro(
                    wallet_addr,
                    cfg=cfg,
                    max_pages=int(max_api_pages),
                    is_benchmark_wallet=is_bench,
                    fast_short_circuit=bool(fast_mode),
                    cached_full_df=cached_df,
                )

            completed_cnt = 0
            with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
                future_to_w = {executor.submit(worker_task, w): w for w in cand_list}
                for future in as_completed(future_to_w):
                    w = future_to_w[future]
                    completed_cnt += 1
                    prog.progress(
                        completed_cnt / len(cand_list),
                        text=f"Đã kiểm toán {completed_cnt}/{len(cand_list)} ví ({time.perf_counter() - t0:.1f}s)...",
                    )
                    try:
                        c_summary, c_pos_df = future.result()
                        results.append(c_summary)
                        if not c_pos_df.empty:
                            details_map[w] = c_pos_df
                    except Exception as e:
                        results.append(make_empty_summary(w, w == ref_wallet_input, f"Lỗi API: {e}"))

            prog.empty()
            res_df = pd.DataFrame(results).sort_values(
                ["passed_filter", "score", "cap_edge", "net_pnl"], ascending=[False, False, False, False]
            ).reset_index(drop=True)
            st.session_state["audit_results"] = res_df
            st.session_state["audit_details"] = details_map
            st.session_state["last_elapsed_sec"] = round(time.perf_counter() - t0, 2)

    if st.session_state["seed_logs"]:
        for lg in st.session_state["seed_logs"]:
            st.caption(f"• {lg}")

    canary = st.session_state["canary_status"]
    if canary is not None:
        status_code, msg = canary
        if status_code == "FOUND":
            st.success(msg)
        else:
            st.info(msg)

    if not st.session_state["seed_preview"].empty:
        with st.expander(f"📋 Xem danh sách {len(st.session_state['seed_preview'])} ví hạt giống thu thập tự nhiên ở Tầng 0", expanded=False):
            st.dataframe(st.session_state["seed_preview"], use_container_width=True, hide_index=True)

    res_df = st.session_state["audit_results"]
    if not res_df.empty:
        st.divider()
        pass_cnt = int(res_df["passed_filter"].sum())
        watch_cnt = int((~res_df["passed_filter"] & (res_df["score"] >= 60)).sum())
        elapsed_str = f"{st.session_state['last_elapsed_sec']}s" if st.session_state["last_elapsed_sec"] else "—"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ví đã kiểm toán", f"{len(res_df)} ví", f"Thời gian: {elapsed_str}")
        k2.metric("✅ Đạt chuẩn Copy-Trade", f"{pass_cnt} ví")
        k3.metric("🟡 Cận chuẩn", f"{watch_cnt} ví")
        k4.metric("Điểm cao nhất", f"{res_df['score'].max()}/100")

        res_df["profile_url"] = "https://polymarket.com/profile/" + res_df["wallet"]

        disp_cols = [
            "verdict", "score", "wallet", "markets", "true_cost", "net_pnl", "true_roi",
            "cap_wr", "cap_edge", "z_score", "hold_pnl_share", "core_band_share",
            "pnl_ex_top5", "med_size", "med_accum_h", "red_flags", "profile_url"
        ]

        st.dataframe(
            res_df[disp_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "verdict": st.column_config.TextColumn("Kết luận", width="medium"),
                "score": st.column_config.ProgressColumn("Điểm Copy", min_value=0, max_value=100, format="%d/100"),
                "wallet": st.column_config.TextColumn("Địa chỉ ví", width="medium"),
                "markets": st.column_config.NumberColumn("Số kèo", format="%d"),
                "true_cost": st.column_config.NumberColumn("Vốn 1 chiều", format="$%,.0f"),
                "net_pnl": st.column_config.NumberColumn("Net PnL", format="$%+,.0f"),
                "true_roi": st.column_config.NumberColumn("True ROI", format="%+.1f%%"),
                "cap_wr": st.column_config.NumberColumn("CapWR", format="%.1f%%"),
                "cap_edge": st.column_config.NumberColumn("CapEdge", format="%+.1f%%"),
                "z_score": st.column_config.NumberColumn("Z-score", format="%.2f"),
                "hold_pnl_share": st.column_config.NumberColumn("% Lãi REDEEM", format="%.0f%%"),
                "core_band_share": st.column_config.NumberColumn("% Vốn 20-85%", format="%.0f%%"),
                "pnl_ex_top5": st.column_config.NumberColumn("PnL ex-Top 5", format="$%+,.0f"),
                "med_size": st.column_config.NumberColumn("Size TV", format="$%.0f"),
                "med_accum_h": st.column_config.NumberColumn("Gom lệnh", format="%.1fh"),
                "red_flags": st.column_config.TextColumn("Chi tiết kiểm toán", width="large"),
                "profile_url": st.column_config.LinkColumn("Polymarket", display_text="Mở Profile"),
            },
        )

        csv_out = res_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Tải kết quả sàng lọc tự nhiên (CSV)",
            data=csv_out,
            file_name=f"polymarket_organic_pro_wallets_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


with tab_xray:
    audited_wallets = list(st.session_state["audit_details"].keys())
    col_x1, col_x2 = st.columns([2, 2])

    selected_from_list = col_x1.selectbox(
        "Chọn ví từ tập đã qua Chốt chặn 2:",
        options=["-- Nhập thủ công bên phải --"] + audited_wallets,
    )
    manual_xray_wallet = col_x2.text_input(
        "Hoặc dán trực tiếp 1 địa chỉ ví `0x...` bất kỳ để giải phẫu sâu:",
        value="",
        placeholder="0x...",
    )

    target_xray = (
        selected_from_list
        if selected_from_list != "-- Nhập thủ công bên phải --"
        else manual_xray_wallet.strip().lower()
    )

    if st.button("🔬 Phân tích X-Ray đầy đủ ví này (Bỏ qua ngắt mạch sớm)", use_container_width=True) and target_xray.startswith("0x"):
        with st.spinner(f"Đang giải phẫu toàn bộ sổ cái của `{target_xray}`..."):
            _, df_one = evaluate_wallet_pro(
                target_xray, cfg=cfg, max_pages=max(int(max_api_pages), 20), fast_short_circuit=False
            )
            if not df_one.empty:
                st.session_state["audit_details"][target_xray] = df_one

    if target_xray in st.session_state["audit_details"]:
        pos_x = st.session_state["audit_details"][target_xray]
        total_c = float(pos_x["true_cost"].sum())
        total_p = float(pos_x["total_pnl"].sum())

        st.subheader(f"Hồ sơ kiểm toán: `{target_xray}`")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Vốn mua 1 chiều", f"${total_c:,.2f}", f"{len(pos_x)} vị thế")
        m2.metric("Net PnL (gồm OPEN)", f"${total_p:+,.2f}", f"True ROI: {total_p/total_c*100 if total_c>0 else 0:+.1f}%")
        m3.metric("Count WR / CapWR", f"{pos_x['real_win'].mean()*100:.1f}%", f"CapWR: {pos_x.loc[pos_x['real_win'], 'true_cost'].sum()/total_c*100 if total_c>0 else 0:.1f}%")
        m4.metric("Size trung vị / Max", f"${pos_x['true_cost'].median():,.1f}", f"Max: ${pos_x['true_cost'].max():,.0f}")
        m5.metric("Gom lệnh / Giữ TB", f"{pos_x['accum_hours'].median():.1f}h", f"Hold: {pos_x['hold_hours'].median():.1f}h")

        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("**1. Phân rã theo Phong cách thoát lệnh (`exit_style`)**")
            exit_rows = []
            for st_name, g in pos_x.groupby("exit_style"):
                gc = g["true_cost"].sum()
                gp = g["total_pnl"].sum()
                exit_rows.append({
                    "Phong cách": st_name,
                    "Số kèo": len(g),
                    "Win Rate": f"{g['real_win'].mean()*100:.1f}%",
                    "Vốn ($)": round(gc, 2),
                    "% Vốn": f"{gc/total_c*100 if total_c>0 else 0:.1f}%",
                    "PnL ($)": round(gp, 2),
                    "ROI": f"{gp/gc*100 if gc>0 else 0:+.1f}%",
                    "Hold TV (h)": round(g["hold_hours"].median(), 1),
                })
            st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

        with c_right:
            st.markdown("**2. Phân bổ vốn & Edge theo Dải xác suất**")
            bins = [0.0, 0.10, 0.20, 0.50, 0.60, 0.85, 0.95, 1.01]
            labels = ["<10% (Penny)", "10–20%", "20–50%", "50–60%", "60–85%", "85–95%", ">=95%"]
            pos_x["band"] = pd.cut(pos_x["avg_price"], bins=bins, labels=labels, right=False)
            band_rows = []
            for b_name, g in pos_x.groupby("band", observed=False):
                if len(g) == 0:
                    continue
                gc = g["true_cost"].sum()
                gp = g["total_pnl"].sum()
                band_rows.append({
                    "Dải giá": str(b_name),
                    "Số kèo": len(g),
                    "Win Rate": f"{g['real_win'].mean()*100:.1f}%",
                    "Vốn ($)": round(gc, 2),
                    "% Vốn": f"{gc/total_c*100 if total_c>0 else 0:.1f}%",
                    "PnL ($)": round(gp, 2),
                    "ROI": f"{gp/gc*100 if gc>0 else 0:+.1f}%",
                })
            st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

        st.markdown("**3. Kỷ luật đi tiền theo Quy mô lệnh của chính ví đó (Conviction Sizing)**")
        q50 = pos_x["true_cost"].quantile(0.50)
        q80 = pos_x["true_cost"].quantile(0.80)
        pos_x["size_tier"] = np.where(
            pos_x["true_cost"] >= q80,
            f"Top 20% Kèo Lớn (>= ${q80:,.1f})",
            np.where(pos_x["true_cost"] > q50, f"Nhóm Trung Bình (${q50:,.1f} – ${q80:,.1f})", f"Bottom 50% Kèo Nhỏ (<= ${q50:,.1f})"),
        )
        tier_rows = []
        for t_name, g in pos_x.groupby("size_tier"):
            gc = g["true_cost"].sum()
            gp = g["total_pnl"].sum()
            tier_rows.append({
                "Phân khúc quy mô": t_name,
                "Số kèo": len(g),
                "Win Rate": f"{g['real_win'].mean()*100:.1f}%",
                "Tổng vốn ($)": round(gc, 2),
                "PnL ($)": round(gp, 2),
                "ROI": f"{gp/gc*100 if gc>0 else 0:+.1f}%",
            })
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

        export_cols = [
            "entry_time_vn", "last_time_vn", "status", "condition_id", "token_id",
            "title", "outcome", "is_hedged", "avg_price", "true_cost", "total_pnl",
            "roi_pct", "real_win", "buy_count", "sell_count", "accum_hours", "hold_hours", "exit_style", "url"
        ]
        avail_export = [c for c in export_cols if c in pos_x.columns]
        st.download_button(
            f"📥 Xuất toàn bộ lịch sử sạch của `{target_xray[:10]}...` (CSV)",
            data=pos_x[avail_export].to_csv(index=False).encode("utf-8-sig"),
            file_name=f"wallet_audit_{target_xray[:10]}.csv",
            mime="text/csv",
        )
