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
    page_title="Polymarket Pro Wallet Finder (Leaderboard Engine)",
    page_icon="🏆",
    layout="wide",
)

LEADERBOARD_V2_URL = "https://data-api.polymarket.com/v2/leaderboard"
BIGGEST_WINNERS_V2_URL = "https://data-api.polymarket.com/v2/biggest-winners"
LEADERBOARD_V1_URL = "https://data-api.polymarket.com/v1/leaderboard"
POSITIONS_V2_URL = "https://data-api.polymarket.com/v2/positions"
ACTIVITY_V2_URL = "https://data-api.polymarket.com/v2/activity"
USER_STATS_V2_URL = "https://data-api.polymarket.com/v2/user-stats"
USER_VOLUME_V2_URL = "https://data-api.polymarket.com/v2/user-volume"
USER_PNL_V2_URL = "https://data-api.polymarket.com/v2/user-pnl"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
TRADES_V2_URL = "https://data-api.polymarket.com/v2/trades"

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


def fetch_v2_paginated(url: str, wallet: str, extra_params: dict | None = None, max_pages: int = 25) -> list[dict]:
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
# 2. TẦNG 0: THU THẬP VÍ HẠT GIỐNG TỪ LEADERBOARD V2 (CHUẨN CURSOR)
# ==============================================================================

def fetch_single_leaderboard_board(
    endpoint: str, category: str, time_period: str, target_count: int
) -> tuple[list[dict], str]:
    """Kéo 1 bảng xếp hạng từ /v2/leaderboard hoặc /v2/biggest-winners bằng phân trang cursor."""
    rows: list[dict] = []
    cursor = None
    max_pages = max(1, math.ceil(target_count / 50))

    tp_v2 = time_period.lower()
    cat_v2 = category.upper()

    for page_idx in range(max_pages):
        if len(rows) >= target_count:
            break

        if cursor is None:
            params = {
                "category": cat_v2,
                "time_period": tp_v2,
                "order_by": "PNL",
                "limit": min(50, target_count - len(rows)),
            }
        else:
            params = {"cursor": cursor}

        try:
            payload = request_with_retry(endpoint, params=params)
        except Exception as e:
            if page_idx == 0:
                return [], f"Lỗi {endpoint}: {e}"
            break

        if isinstance(payload, dict):
            batch = payload.get("data") or payload.get("results") or []
            pagination = payload.get("pagination") or {}
            cursor = pagination.get("next_cursor")
            has_more = pagination.get("has_more", bool(cursor))
        elif isinstance(payload, list):
            batch = payload
            cursor = None
            has_more = False
        else:
            break

        if not batch:
            break

        for item in batch:
            if not isinstance(item, dict):
                continue
            w = str(
                item.get("proxy_wallet")
                or item.get("proxyWallet")
                or item.get("user")
                or item.get("address")
                or ""
            ).strip().lower()
            if not w.startswith("0x") or len(w) != 42:
                continue

            rank_val = item.get("rank") or (len(rows) + 1)
            rows.append({
                "wallet": w,
                "userName": item.get("user_name") or item.get("userName") or item.get("name") or "",
                "source_board": f"{cat_v2}/{tp_v2}",
                "lb_rank": int(rank_val),
                "lb_pnl": float(item.get("pnl") or item.get("total_pnl") or 0),
                "lb_vol": float(item.get("vol") or item.get("volume") or item.get("total_volume") or 0),
            })

        if not cursor or not has_more:
            break

    return rows, ""


def seed_wallets_from_leaderboard_multi(
    categories: list[str], time_periods: list[str], limit_per_board: int, include_biggest_winners: bool = True
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """
    Tầng 0A (Chủ lực): Đào ví tự nhiên từ Data API /v2/leaderboard (và /v2/biggest-winners)
    qua nhiều Category & khung thời gian, giữ thứ hạng cao nhất của từng ví.
    """
    logs: list[str] = []
    all_rows: list[dict] = []

    for cat in categories:
        for tp in time_periods:
            rows, err = fetch_single_leaderboard_board(LEADERBOARD_V2_URL, cat, tp, limit_per_board)
            if err and not rows:
                v1_rows = []
                offset = 0
                while len(v1_rows) < limit_per_board:
                    try:
                        p_v1 = request_with_retry(
                            LEADERBOARD_V1_URL,
                            {
                                "category": cat.upper(),
                                "timePeriod": tp.upper(),
                                "orderBy": "PNL",
                                "limit": min(50, limit_per_board - len(v1_rows)),
                                "offset": offset,
                            },
                        )
                        b_v1 = p_v1.get("data") or [] if isinstance(p_v1, dict) else (p_v1 or [])
                        if not b_v1:
                            break
                        for item in b_v1:
                            w = str(item.get("proxyWallet") or item.get("proxy_wallet") or item.get("user") or "").strip().lower()
                            if w.startswith("0x") and len(w) == 42:
                                v1_rows.append({
                                    "wallet": w,
                                    "userName": item.get("userName") or item.get("user_name") or "",
                                    "source_board": f"{cat}/{tp}",
                                    "lb_rank": len(v1_rows) + 1,
                                    "lb_pnl": float(item.get("pnl") or 0),
                                    "lb_vol": float(item.get("vol") or item.get("volume") or 0),
                                })
                        offset += 50
                        if len(b_v1) < 50:
                            break
                    except Exception as e_v1:
                        logs.append(f"Cảnh báo Leaderboard `{cat}/{tp}`: {err} | {e_v1}")
                        break
                rows = v1_rows

            if rows:
                logs.append(f"Leaderboard `{cat}/{tp}`: Kéo được **{len(rows)}** ví.")
                all_rows.extend(rows)

            if include_biggest_winners:
                bw_rows, _ = fetch_single_leaderboard_board(BIGGEST_WINNERS_V2_URL, cat, tp, min(50, limit_per_board))
                if bw_rows:
                    for r in bw_rows:
                        r["source_board"] = f"Winners:{cat}/{tp}"
                    all_rows.extend(bw_rows)

    if not all_rows:
        return [], pd.DataFrame(), logs

    df = pd.DataFrame(all_rows)
    df = (
        df.sort_values(["lb_rank", "lb_pnl"], ascending=[True, False])
        .drop_duplicates(subset=["wallet"], keep="first")
        .reset_index(drop=True)
    )
    logs.append(f"Tổng hợp sau khử trùng lặp: **{len(df)}** ví độc lập từ Leaderboard.")
    return df["wallet"].tolist(), df, logs


def seed_wallets_from_markets(
    category_filter: str, num_markets: int, min_accum_per_market: float
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Tầng 0B (Phụ trợ): Đào ví từ Gamma API (/events) + Data API v2 (/v2/trades)."""
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
            logs.append(f"Cảnh báo Gamma /events: {e}")
            break

    kw_map = {
        "Esports (CS2 / Dota / LoL / Valorant)": ["counter-strike", "cs2", "dota", "league of legends", "valorant", "esports"],
        "Sports Truyền Thống (NBA / Tennis / Bóng đá / UFC)": ["nba", "nfl", "nhl", "mlb", "atp", "wta", "tennis", "premier league", "champions league", "la liga", "soccer", "ufc"],
        "Finance (Chỉ số / Cổ phiếu / Vàng / Lãi suất)": ["s&p", "nasdaq", "dow jones", "nvda", "tsla", "aapl", "gold", "oil", "fed", "cpi", "treasury", "close above"],
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
        if any(k in full_text for k in kw_map.get(category_filter, [])):
            seen_cids.add(cid)
            filtered_markets.append(m)

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

    with ThreadPoolExecutor(max_workers=6) as pool:
        for cid, title, trades_batch in pool.map(fetch_market_trades, target_markets):
            cid_title_map[cid] = title
            for t in trades_batch:
                if not isinstance(t, dict):
                    continue
                w = str(t.get("proxy_wallet") or t.get("proxyWallet") or t.get("user") or "").strip().lower()
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
            {"wallet": w, "qualifying_markets": 0, "total_clips": 0, "seed_usdc": 0.0, "sample_market": cid_title_map.get(cid, "")},
        )
        rec["qualifying_markets"] += 1
        rec["total_clips"] += pair_clips[(w, cid)]
        rec["seed_usdc"] += tot_usdc

    seed_df = pd.DataFrame(list(wallet_agg.values()))
    if not seed_df.empty:
        seed_df = seed_df.sort_values(["seed_usdc", "total_clips"], ascending=[False, False]).reset_index(drop=True)
        logs.append(f"Market Trades: Thu thập được **{len(seed_df)}** ví.")
        return seed_df["wallet"].tolist(), seed_df, logs
    return [], pd.DataFrame(), logs


# ==============================================================================
# 3. KIỂM TOÁN 3 CHẶN TIẾN TIẾN (FAST SURFACE -> POSITIONS -> DEEP ACTIVITY)
# ==============================================================================

def fast_surface_check(wallet: str, cfg: dict) -> tuple[bool, str, dict]:
    """Chặng 1 (~0.25s): Loại nhanh ví mới, bot MM quá rộng, hoặc Maker 99¢ biên lãi siêu mỏng."""
    w = wallet.strip().lower()
    stats_data, vol_data, pnl_data = {}, {}, {}

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

    if m_traded is not None and m_traded > 0:
        if m_traded < cfg["min_markets"] * 0.8:
            return False, f"⚡ Chặn sớm Tầng 1: Ít mẫu ({int(m_traded)} < {cfg['min_markets']} kèo)", meta
        if m_traded > cfg["max_markets"] * 1.25:
            return False, f"⚡ Chặn sớm Tầng 1: Đánh quá rộng ({int(m_traded)} > {cfg['max_markets']} kèo, nghi bot MM)", meta

    if ui_pnl is not None and ui_pnl < cfg["min_net_pnl"] * 0.7:
        return False, f"⚡ Chặn sớm Tầng 1: Net PnL thấp (${ui_pnl:,.0f} < ${cfg['min_net_pnl']:,.0f})", meta

    if ui_pnl is not None and ui_vol is not None and ui_vol > 1000:
        surf_roi = (ui_pnl / ui_vol) * 100.0
        if surf_roi < cfg["min_true_roi"] * 0.35:
            return False, f"⚡ Chặn sớm Tầng 1: PnL/Volume bề mặt quá mỏng ({surf_roi:+.2f}% — nghi Maker 99¢)", meta

    return True, "", meta


def fetch_positions_only_df(wallet: str, max_pages: int = 20) -> pd.DataFrame:
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


def enrich_with_activity_df(wallet: str, pos_df: pd.DataFrame, max_act_pages: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chặng 2B: Kéo sâu /v2/activity (tới 15,000 giao dịch gần nhất cho các ví ứng viên vượt qua Chặng 2A)
    và khôi phục vốn mua chuẩn 1 chiều (kể cả các vị thế CLOSED cũ nằm ngoài cửa sổ activity).
    """
    def get_col(df: pd.DataFrame, names: list[str], default=None) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df), index=df.index)

    raw_act: list[dict] = []
    try:
        raw_act = fetch_v2_paginated(ACTIVITY_V2_URL, wallet, {}, max_pages=max_act_pages)
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

    # Khôi phục vốn mua 1 chiều chuẩn:
    # 1) Ưu tiên act_buy_usdc từ /v2/activity
    # 2) Fallback entry_cost_usdc từ /v2/positions
    # 3) Với các vị thế CLOSED cũ nằm ngoài cửa sổ activity (entry_cost_usdc == 0), ước tính ngược từ realized_pnl & avg_price
    raw_cost = np.where(out_df["act_buy_usdc"] > 0, out_df["act_buy_usdc"], out_df["entry_cost_usdc"])
    missing_mask = (raw_cost <= 0) & (out_df["avg_price"] > 0.005) & (out_df["avg_price"] < 0.995)
    implied_cost = np.where(
        out_df["realized_pnl"] > 0,
        out_df["realized_pnl"] * out_df["avg_price"] / (1.0 - out_df["avg_price"]),
        0.0,
    )
    out_df["true_cost"] = np.where(missing_mask, implied_cost, raw_cost)
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
    max_pages: int = 20,
    is_benchmark_wallet: bool = False,
    fast_short_circuit: bool = True,
    cached_full_df: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Pipeline kiểm toán 3 chặng có ngắt mạch sớm cho ví nhiễu & kéo sâu activity cho ví ứng viên tiềm năng."""
    if cached_full_df is not None and not cached_full_df.empty and cached_full_df.get("_activity_enriched", pd.Series([False])).iloc[0]:
        pos_df = cached_full_df
    else:
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

        raw_pos_df = fetch_positions_only_df(wallet, max_pages=max_pages)
        if raw_pos_df.empty:
            return make_empty_summary(wallet, is_benchmark_wallet, "Ví không có lịch sử vị thế"), pd.DataFrame()

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

            price_weights = np.where(raw_pos_df["entry_cost_usdc"] > 0, raw_pos_df["entry_cost_usdc"], 1.0)
            tot_w = float(np.sum(price_weights))
            fav_w = float(np.sum(price_weights[raw_pos_df["avg_price"] >= 0.95])) / tot_w * 100.0
            if fav_w > cfg["max_high_fav_share"] * 1.3:
                return make_empty_summary(
                    wallet, is_benchmark_wallet,
                    f"⚡ Dừng sau /positions: Nghi bẫy Maker/Fav >=95¢ ({fav_w:.0f}%)",
                    markets=n_mkt_pre, net_pnl=pnl_pre, true_cost=cost_pre,
                ), pd.DataFrame()

        pos_df, _ = enrich_with_activity_df(wallet, raw_pos_df, max_act_pages=max(max_pages, 25))

    n_markets = int(pos_df["condition_id"].replace("", np.nan).nunique())
    if n_markets == 0:
        n_markets = len(pos_df)
    n_pos = len(pos_df)

    total_cost = float(pos_df["true_cost"].sum())
    net_pnl = float(pos_df["total_pnl"].sum())
    true_roi = (net_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

    cost_valid_df = pos_df[pos_df["true_cost"] > 0]
    valid_cost_sum = float(cost_valid_df["true_cost"].sum())

    count_wr = float(pos_df["real_win"].mean() * 100.0)
    if valid_cost_sum > 0:
        cap_wr = float(cost_valid_df.loc[cost_valid_df["real_win"], "true_cost"].sum() / valid_cost_sum * 100.0)
        w_entry = float((cost_valid_df["avg_price"] * cost_valid_df["true_cost"]).sum() / valid_cost_sum * 100.0)
    else:
        cap_wr = count_wr
        w_entry = float(pos_df["avg_price"].mean() * 100.0)
    cap_edge = cap_wr - w_entry

    if valid_cost_sum > 0 and len(cost_valid_df) >= 10:
        weights = cost_valid_df["true_cost"] / valid_cost_sum
        probs = cost_valid_df["avg_price"].clip(0.005, 0.995)
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

    core_band_cost = float(cost_valid_df[(cost_valid_df["avg_price"] >= 0.20) & (cost_valid_df["avg_price"] <= 0.85)]["true_cost"].sum())
    penny_cost = float(cost_valid_df[cost_valid_df["avg_price"] < 0.10]["true_cost"].sum())
    high_fav_cost = float(cost_valid_df[cost_valid_df["avg_price"] >= 0.95]["true_cost"].sum())

    core_band_share = (core_band_cost / valid_cost_sum * 100.0) if valid_cost_sum > 0 else 0.0
    penny_share = (penny_cost / valid_cost_sum * 100.0) if valid_cost_sum > 0 else 0.0
    high_fav_share = (high_fav_cost / valid_cost_sum * 100.0) if valid_cost_sum > 0 else 0.0

    top5_wins = float(pos_df[pos_df["total_pnl"] > 0].sort_values("total_pnl", ascending=False)["total_pnl"].head(5).sum())
    pnl_ex_top5 = net_pnl - top5_wins

    if not cost_valid_df.empty:
        q80 = cost_valid_df["true_cost"].quantile(0.80)
        q50 = cost_valid_df["true_cost"].quantile(0.50)
        top20_df = cost_valid_df[cost_valid_df["true_cost"] >= q80]
        bot50_df = cost_valid_df[cost_valid_df["true_cost"] <= q50]
        top20_roi = (top20_df["total_pnl"].sum() / top20_df["true_cost"].sum() * 100.0) if top20_df["true_cost"].sum() > 0 else 0.0
        bot50_roi = (bot50_df["total_pnl"].sum() / bot50_df["true_cost"].sum() * 100.0) if bot50_df["true_cost"].sum() > 0 else 0.0
        med_size = float(cost_valid_df["true_cost"].median())
    else:
        top20_roi, bot50_roi, med_size = 0.0, 0.0, 0.0

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
# 4. STREAMLIT UI (LEADERBOARD-FIRST ENGINE)
# ==============================================================================

st.title("🏆 Trạm Săn & Kiểm Toán Ví Pro Polymarket (Leaderboard v2 Engine)")
st.caption(
    "Ưu tiên đào từ **Leaderboard Data API v2 (`/v2/leaderboard` + `/v2/biggest-winners`)** để bắt trọn các ví gom lệnh "
    "pre-event có PnL tích lũy cao (như ví CS2 mẫu), sau đó sàng lọc 3 chặng để loại bỏ bẫy Maker 99¢ và Scalper bán sớm."
)

with st.sidebar:
    st.header("🎯 1. Ví Mẫu Kiểm Định (Canary)")
    ref_wallet_input = st.text_input(
        "Địa chỉ Ví Mẫu dùng kiểm tra độ phủ Leaderboard",
        value=DEFAULT_BENCHMARK_WALLET,
    ).strip().lower()

    st.divider()
    st.header("⚡ 2. Tốc độ & Độ sâu Sổ cái")
    c_spd1, c_spd2 = st.columns(2)
    max_api_pages = c_spd1.number_input("Trang /positions (×500)", min_value=5, max_value=40, value=15, step=5)
    num_workers = c_spd2.number_input("Luồng song song", min_value=2, max_value=10, value=6, step=1)
    fast_mode = st.checkbox(
        "Chặn sớm 3 chặng (Bỏ qua /activity với ví rớt ngưỡng)",
        value=True,
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

tab_scan, tab_xray = st.tabs(["🏆 1. Đào từ Leaderboard & Sàng Lọc", "🔬 2. Soi Sâu Từng Ví (X-Ray & Xuất CSV)"])

with tab_scan:
    seed_mode = st.radio(
        "Chọn nguồn thu thập ví hạt giống:",
        options=[
            "Nguồn 1 (Khuyên dùng): Đào từ Leaderboard v2 (/v2/leaderboard)",
            "Nguồn 2: Nhập danh sách ví thủ công / Upload CSV",
            "Nguồn 3: Đào từ Market thanh khoản cao (/v2/trades)",
        ],
        horizontal=True,
    )

    manual_candidates: list[str] = []

    if seed_mode.startswith("Nguồn 1"):
        col_b1, col_b2, col_b3, col_b4 = st.columns([2, 1.5, 1, 1])
        lb_cats = col_b1.multiselect(
            "Category Leaderboard (có thể chọn nhiều)",
            options=["SPORTS", "FINANCE", "POLITICS", "ECONOMICS", "TECH", "OVERALL"],
            default=["SPORTS", "FINANCE"],
        )
        lb_periods = col_b2.multiselect(
            "Khung thời gian (time_period)",
            options=["all", "month", "week"],
            default=["all", "month"],
        )
        lb_limit = col_b3.number_input("Top N ví / bảng", 25, 250, 100, 25)
        inc_bw = col_b4.checkbox("Gộp thêm Biggest Winners", value=True)

    elif seed_mode.startswith("Nguồn 2"):
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

    else:
        col_a1, col_a2, col_a3, col_a4 = st.columns([2.2, 1, 1.1, 1])
        cat_choice = col_a1.selectbox(
            "Nhóm thị trường mục tiêu",
            [
                "Esports (CS2 / Dota / LoL / Valorant)",
                "Sports Truyền Thống (NBA / Tennis / Bóng đá / UFC)",
                "Finance (Chỉ số / Cổ phiếu / Vàng / Lãi suất)",
            ],
        )
        n_mkts = col_a2.number_input("Số trận quét", 5, 80, 30, 5)
        min_accum_trd = col_a3.number_input("Tổng gom tối thiểu / trận ($)", 50, 20000, 1000, 100)
        max_cand_a = col_a4.number_input("Top N ví kiểm toán", 5, 100, 30, 5)

    run_scan_clicked = st.button(
        "🚀 ĐÀO TỪ LEADERBOARD & CHẠY KIỂM TOÁN 3 CHẶNG",
        type="primary",
        use_container_width=True,
    )

    if run_scan_clicked:
        t0 = time.perf_counter()

        if seed_mode.startswith("Nguồn 1"):
            if not lb_cats or not lb_periods:
                st.warning("Vui lòng chọn ít nhất 1 Category và 1 khung thời gian.")
                cand_list, raw_seeded = [], []
            else:
                with st.spinner("Đang kéo danh sách ví từ Polymarket Leaderboard v2 (cursor pagination)..."):
                    raw_seeded, seed_df, logs = seed_wallets_from_leaderboard_multi(
                        lb_cats, lb_periods, int(lb_limit), include_biggest_winners=bool(inc_bw)
                    )
                    cand_list = raw_seeded
                    st.session_state["seed_preview"] = seed_df
                    st.session_state["seed_logs"] = logs
        elif seed_mode.startswith("Nguồn 2"):
            raw_seeded = manual_candidates
            cand_list = manual_candidates
            st.session_state["seed_preview"] = pd.DataFrame()
            st.session_state["seed_logs"] = [f"Nguồn thủ công: Nhận **{len(cand_list)}** địa chỉ ví."]
        else:
            with st.spinner("Đang quét sổ lệnh Gamma API & /v2/trades..."):
                raw_seeded, seed_df, logs = seed_wallets_from_markets(cat_choice, int(n_mkts), float(min_accum_trd))
                cand_list = raw_seeded[: int(max_cand_a)]
                st.session_state["seed_preview"] = seed_df
                st.session_state["seed_logs"] = logs

        if cand_list:
            if ref_wallet_input in raw_seeded:
                rank_in_seed = raw_seeded.index(ref_wallet_input) + 1
                st.session_state["canary_status"] = (
                    "FOUND",
                    f"🎯 **ĐÃ BẮT TRÚNG VÍ MẪU TRÊN LEADERBOARD:** `{ref_wallet_input}` xuất hiện tự nhiên ở **vị trí #{rank_in_seed}/{len(raw_seeded)}** "
                    "và đang tự đi qua các tầng kiểm toán bên dưới!",
                )
            else:
                st.session_state["canary_status"] = (
                    "NOT_FOUND",
                    f"ℹ️ **Ví Mẫu `{ref_wallet_input[:10]}...` chưa nằm trong danh sách vừa kéo** "
                    "(nếu đang quét `SPORTS / all`, hãy tăng 'Top N ví / bảng' lên `150–200` để chạm tới đúng thứ hạng của ví mẫu).",
                )

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
        with st.expander(f"📋 Xem danh sách {len(st.session_state['seed_preview'])} ví thu thập từ Leaderboard ở Tầng 0", expanded=False):
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
            "📥 Tải kết quả sàng lọc Leaderboard (CSV)",
            data=csv_out,
            file_name=f"polymarket_leaderboard_pro_wallets_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M')}.csv",
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
                target_xray, cfg=cfg, max_pages=max(int(max_api_pages), 25), fast_short_circuit=False
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
