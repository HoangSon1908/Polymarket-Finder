import math
import re
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Polymarket Pro Wallet Finder",
    page_icon="🎯",
    layout="wide",
)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
TRADES_V2_URL = "https://data-api.polymarket.com/v2/trades"
LEADERBOARD_V2_URL = "https://data-api.polymarket.com/v2/leaderboard"
LEADERBOARD_V1_URL = "https://data-api.polymarket.com/v1/leaderboard"
POSITIONS_V2_URL = "https://data-api.polymarket.com/v2/positions"
ACTIVITY_V2_URL = "https://data-api.polymarket.com/v2/activity"

BENCHMARK_CS2_WALLET = "0xc9bfc7986767173b4f1c3fc9b2c6b255fc7381c0"
VN_TZ = timezone(timedelta(hours=7))


# ==============================================================================
# 1. API HELPERS (CHUẨN DATA API V2 & GAMMA API)
# ==============================================================================

def request_with_retry(url: str, params: dict, retries: int = 3, timeout: int = 15):
    last_err = None
    for attempt in range(retries):
        try:
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            res.raise_for_status()
            return res.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Lỗi gọi {url}: {last_err}")


def fetch_v2_paginated(url: str, wallet: str, extra_params: dict | None = None, max_pages: int = 25) -> list[dict]:
    """Kéo dữ liệu chuẩn Polymarket Data API v2 bằng phân trang cursor."""
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


def seed_wallets_from_markets(
    category_filter: str, num_markets: int, min_trade_usdc: float
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """
    Tầng 0A: Đào ví hạt giống từ Gamma API (/events + /markets) kết hợp Data API v2 (/v2/trades).
    Trả về (danh_sách_ví, bảng_chi_tiết, danh_sách_cảnh_báo_lỗi).
    """
    logs: list[str] = []

    # Kéo cả /events lẫn /markets để lấy đầy đủ sự kiện Sports & Finance thanh khoản cao
    raw_markets: list[dict] = []
    try:
        ev_list = request_with_retry(
            GAMMA_EVENTS_URL,
            {
                "limit": 100,
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
    except Exception as e:
        logs.append(f"Cảnh báo Gamma /events: {e}")

    if not raw_markets:
        try:
            mkt_list = request_with_retry(
                GAMMA_MARKETS_URL,
                {
                    "limit": 150,
                    "order": "volume24hr",
                    "ascending": "false",
                    "active": "true",
                    "closed": "false",
                },
            ) or []
            for m in mkt_list:
                evs = m.get("events") or []
                ev0 = evs[0] if isinstance(evs, list) and evs else {}
                m_copy = dict(m)
                m_copy["_ev_title"] = str(ev0.get("title") or "")
                m_copy["_ev_slug"] = str(ev0.get("slug") or "")
                m_copy["_ev_cat"] = str(m.get("category") or "")
                raw_markets.append(m_copy)
        except Exception as e:
            logs.append(f"Lỗi Gamma /markets: {e}")
            return [], pd.DataFrame(), logs

    kw_map = {
        "Sports (NBA / Tennis / Bóng đá / Esports)": [
            "nba", "nfl", "nhl", "mlb", "ncaa", "atp", "wta", "tennis", "premier league",
            "champions league", "la liga", "serie a", "bundesliga", "ligue 1", "soccer",
            "football", "counter-strike", "cs2", "dota", "league of legends", "lol",
            "valorant", "ufc", "mma", "f1", "formula 1", "vs.", "vs ", "spread:", "o/u "
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

        # Loại bỏ Crypto ngắn hạn (phí 0.07, cuộc chơi độ trễ HFT)
        if any(k in full_text for k in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "up or down"]):
            continue

        if category_filter == "Tất cả (trừ Crypto ngắn hạn)":
            seen_cids.add(cid)
            filtered_markets.append(m)
        else:
            kws = kw_map.get(category_filter, [])
            if any(k in full_text for k in kws):
                seen_cids.add(cid)
                filtered_markets.append(m)

    logs.append(f"Gamma API: Tìm thấy **{len(filtered_markets)}** market khớp nhóm `{category_filter}`.")
    target_markets = filtered_markets[:num_markets]
    wallet_stats: dict[str, dict] = {}
    trade_errors = 0

    for m in target_markets:
        cid = str(m.get("conditionId") or m.get("condition_id") or "").strip()
        title = str(m.get("question") or m.get("_ev_title") or "")
        trades_batch: list[dict] = []

        # Thử tham số chuẩn Data API v2: condition=<cid>, fallback market=<cid>
        for param_key in ("condition", "market"):
            try:
                payload = request_with_retry(
                    TRADES_V2_URL,
                    params={param_key: cid, "limit": 200},
                )
                if isinstance(payload, dict):
                    trades_batch = payload.get("data") or []
                elif isinstance(payload, list):
                    trades_batch = payload
                if trades_batch:
                    break
            except Exception as e:
                if param_key == "market":
                    trade_errors += 1
                    if trade_errors <= 2:
                        logs.append(f"Lỗi `/v2/trades` trên market `{cid[:10]}...`: {e}")

        for t in trades_batch:
            if not isinstance(t, dict):
                continue
            w = str(
                t.get("proxy_wallet")
                or t.get("proxyWallet")
                or t.get("user")
                or t.get("taker_address")
                or t.get("maker_address")
                or ""
            ).strip().lower()
            if not w.startswith("0x") or len(w) != 42:
                continue

            size = float(t.get("size") or t.get("shares") or 0)
            price = float(t.get("price") or 0)
            usdc = float(t.get("usdc_size") or t.get("usdcSize") or (size * price))

            if usdc < min_trade_usdc:
                continue

            rec = wallet_stats.setdefault(
                w,
                {"wallet": w, "seed_trades": 0, "seed_usdc": 0.0, "sample_market": title},
            )
            rec["seed_trades"] += 1
            rec["seed_usdc"] += usdc

    seed_df = pd.DataFrame(list(wallet_stats.values()))
    if not seed_df.empty:
        seed_df = seed_df.sort_values(["seed_trades", "seed_usdc"], ascending=[False, False]).reset_index(drop=True)
        logs.append(f"Data API `/v2/trades`: Thu thập được **{len(seed_df)}** ví độc lập có lệnh >= ${min_trade_usdc:,.0f}.")
        return seed_df["wallet"].tolist(), seed_df, logs

    logs.append("Không tìm thấy ví nào khớp ngưỡng lệnh tối thiểu trong lô market vừa quét (hãy thử giảm ngưỡng 'Lệnh mẫu tối thiểu ($)' xuống $50–$100).")
    return [], pd.DataFrame(), logs


def seed_wallets_from_leaderboard(category: str, time_period: str, limit: int = 50) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Tầng 0B: Lấy danh sách ví từ Leaderboard Polymarket."""
    logs: list[str] = []
    params = {
        "category": category,
        "timePeriod": time_period,
        "orderBy": "PNL",
        "limit": min(limit, 50),
        "offset": 0,
    }
    data = None
    for ep in (LEADERBOARD_V2_URL, LEADERBOARD_V1_URL):
        try:
            payload = request_with_retry(ep, params=params)
            if isinstance(payload, dict):
                data = payload.get("data") or payload.get("results") or []
            elif isinstance(payload, list):
                data = payload
            if data:
                break
        except Exception as e:
            logs.append(f"Thử {ep}: {e}")

    if not data:
        return [], pd.DataFrame(), logs

    rows = []
    for item in data:
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
                "wallet": w,
                "userName": item.get("user_name") or item.get("userName") or item.get("name") or "",
                "lb_pnl": float(item.get("pnl") or 0),
                "lb_vol": float(item.get("vol") or item.get("volume") or 0),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["wallet"]) if rows else pd.DataFrame()
    if not df.empty:
        logs.append(f"Leaderboard: Kéo thành công **{len(df)}** ví.")
    return (df["wallet"].tolist() if not df.empty else []), df, logs


# ==============================================================================
# 2. DEEP LEDGER & COPYABILITY AUDIT ENGINE (TẦNG 1 + TẦNG 2)
# ==============================================================================

def build_wallet_positions_df(wallet: str, max_pages: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kéo toàn bộ vòng đời vị thế từ /v2/positions (4 trạng thái) và đối soát với /v2/activity
    để khử trùng lặp, tính đúng vốn mua thực 1 chiều (act_buy_usdc), hành vi gom/thoát lệnh.
    """
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
        return pd.DataFrame(), pd.DataFrame()

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

    act_records = [act_agg_map.get(tid, {}) for tid in pos_df["token_id"]]
    act_merged = pd.DataFrame(act_records, index=pos_df.index)

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
            pos_df[col] = act_merged[col].fillna(default_v)
        else:
            pos_df[col] = default_v

    pos_df["true_cost"] = np.where(pos_df["act_buy_usdc"] > 0, pos_df["act_buy_usdc"], pos_df["entry_cost_usdc"])
    pos_df["total_pnl"] = pos_df["realized_pnl"] + pos_df["unrealized_pnl"]
    pos_df["roi_pct"] = np.where(pos_df["true_cost"] > 0, pos_df["total_pnl"] / pos_df["true_cost"] * 100.0, 0.0)
    pos_df["real_win"] = pos_df["total_pnl"] > 0

    cid_counts = pos_df[pos_df["condition_id"] != ""].groupby("condition_id")["outcome"].nunique()
    hedged_cids = set(cid_counts[cid_counts >= 2].index)
    pos_df["is_hedged"] = pos_df["condition_id"].isin(hedged_cids)

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

    pos_df["exit_style"] = pos_df.apply(classify_exit, axis=1)
    pos_df["entry_time_vn"] = pd.to_datetime(pos_df["first_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    pos_df["last_time_vn"] = pd.to_datetime(pos_df["last_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)

    return pos_df, act_df


def evaluate_wallet_pro(wallet: str, cfg: dict, max_pages: int = 20) -> tuple[dict, pd.DataFrame]:
    """Chạy full kiểm toán Tầng 1 + Tầng 2 và chấm điểm khả năng Copy-Trade (0-100)."""
    pos_df, _ = build_wallet_positions_df(wallet, max_pages=max_pages)
    if pos_df.empty:
        return {
            "wallet": wallet,
            "verdict": "❌ KHÔNG DỮ LIỆU",
            "score": 0,
            "markets": 0,
            "positions": 0,
            "true_cost": 0.0,
            "net_pnl": 0.0,
            "true_roi": 0.0,
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
            "red_flags": "Ví không có lịch sử vị thế",
        }, pos_df

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
        flags.append(f"Edge/Z-score mức trung bình (Edge {cap_edge:+.1f}%, Z={z_score:.2f})")
    else:
        flags.append(f"Thiếu Edge thống kê (CapEdge {cap_edge:+.1f}%, Z={z_score:.2f})")

    if hold_pnl_share >= cfg["min_hold_pnl_share"]:
        score += 20
    else:
        flags.append(f"Lãi phụ thuộc bán non SOLD_EARLY (Hold chỉ chiếm {hold_pnl_share:.0f}% lãi, Early=${early_pnl:+,.0f})")

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
        net_pnl < cfg["min_net_pnl"]
        or true_roi < cfg["min_true_roi"]
        or hold_pnl_share < cfg["min_hold_pnl_share"]
        or high_fav_share > cfg["max_high_fav_share"]
        or pnl_ex_top5 <= 0
    )

    if score >= 80 and not critical_fail:
        verdict = "✅ ĐẠT CHUẨN COPY"
    elif score >= 60:
        verdict = "🟡 THEO DÕI THÊM"
    else:
        verdict = "❌ LOẠI"

    summary = {
        "wallet": wallet,
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
        "red_flags": " | ".join(flags) if flags else "Sạch lỗi — Đạt toàn bộ DNA ví CS2",
    }
    return summary, pos_df


# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================

st.title("🎯 Trạm Săn & Kiểm Toán Ví Pro Polymarket (Chuẩn Copy-Trade)")
st.caption(
    f"Đối chiếu trực tiếp theo DNA vận hành của ví CS2 chuẩn `{BENCHMARK_CS2_WALLET}`: "
    "Vốn thực 1 chiều • Giữ đến đáo hạn (REDEEM) • Dải giá 20%–85% • Nhồi tiền có Edge thống kê."
)

with st.sidebar:
    st.header("⚙️ Ngưỡng Lọc Chuẩn Ví Pro")

    preset = st.selectbox(
        "Bộ cấu hình mẫu",
        options=[
            "Chuẩn Ví CS2 (0xc9bfc798...)",
            "Mở rộng (Tìm ví tiềm năng vốn vừa)",
            "Khắt khe (Chỉ lấy Super-Pro)",
        ],
        index=0,
    )

    if preset == "Chuẩn Ví CS2 (0xc9bfc798...)":
        def_min_m, def_max_m = 80, 1500
        def_pnl, def_roi = 5000.0, 10.0
        def_edge, def_z = 4.0, 1.65
        def_hold, def_core, def_fav = 80.0, 70.0, 10.0
    elif preset == "Mở rộng (Tìm ví tiềm năng vốn vừa)":
        def_min_m, def_max_m = 50, 2000
        def_pnl, def_roi = 2500.0, 8.0
        def_edge, def_z = 3.0, 1.40
        def_hold, def_core, def_fav = 75.0, 65.0, 15.0
    else:
        def_min_m, def_max_m = 120, 1200
        def_pnl, def_roi = 20000.0, 12.0
        def_edge, def_z = 5.0, 2.00
        def_hold, def_core, def_fav = 85.0, 75.0, 5.0

    c1, c2 = st.columns(2)
    min_markets = c1.number_input("Số kèo tối thiểu", min_value=10, max_value=1000, value=def_min_m, step=10)
    max_markets = c2.number_input("Số kèo tối đa", min_value=100, max_value=5000, value=def_max_m, step=100)

    min_net_pnl = st.number_input("Net PnL tối thiểu ($)", min_value=500.0, value=def_pnl, step=1000.0)
    min_true_roi = st.slider("True ROI tối thiểu trên vốn thực (%)", 2.0, 30.0, def_roi, 0.5)

    st.divider()
    st.subheader("📐 Kiểm định Edge & Hành vi Copy")
    min_cap_edge = st.slider("CapEdge tối thiểu (%)", 1.0, 15.0, def_edge, 0.5)
    min_z_score = st.slider("Z-score thống kê tối thiểu", 1.0, 3.5, def_z, 0.05)
    min_hold_pnl_share = st.slider("Tỷ trọng lãi từ giữ đáo hạn REDEEM (%)", 50.0, 100.0, def_hold, 5.0)
    min_core_band_share = st.slider("Tỷ trọng vốn dải 20%–85% (%)", 40.0, 95.0, def_core, 5.0)
    max_high_fav_share = st.slider("Trần vốn dải >=95¢ — Chặn Maker 99¢ (%)", 0.0, 30.0, def_fav, 1.0)

    max_api_pages = st.slider("Độ sâu kéo lịch sử (số trang × 500)", 5, 50, 20, 5)

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

if "audit_results" not in st.session_state:
    st.session_state["audit_results"] = pd.DataFrame()
if "audit_details" not in st.session_state:
    st.session_state["audit_details"] = {}
if "seed_logs" not in st.session_state:
    st.session_state["seed_logs"] = []

tab_scan, tab_xray = st.tabs(["🔍 1. Đào & Quét Hàng Loạt Ví Pro", "🔬 2. Soi Sâu Từng Ví (X-Ray & Xuất CSV)"])

# ==============================================================================
# TAB 1: BATCH SEEDING & AUDIT (1-CLICK HOẶC TỪNG BƯỚC)
# ==============================================================================
with tab_scan:
    seed_mode = st.radio(
        "Chọn nguồn nạp danh sách ví hạt giống:",
        options=[
            "Nguồn A: Tự đào từ Market thanh khoản cao (Gamma + Data API v2)",
            "Nguồn B: Kéo từ Bảng Xếp Hạng (Leaderboard Polymarket)",
            "Nguồn C: Dán danh sách ví thủ công / Upload CSV",
        ],
        horizontal=True,
    )

    if seed_mode.startswith("Nguồn A"):
        col_a1, col_a2, col_a3, col_a4 = st.columns([2, 1, 1, 1])
        cat_choice = col_a1.selectbox(
            "Nhóm thị trường mục tiêu",
            [
                "Sports (NBA / Tennis / Bóng đá / Esports)",
                "Finance (Chỉ số / Cổ phiếu / Vàng / Lãi suất)",
                "Politics & Macro",
                "Tất cả (trừ Crypto ngắn hạn)",
            ],
        )
        n_mkts = col_a2.number_input("Số market quét", 5, 60, 20, 5)
        min_trd = col_a3.number_input("Lệnh mẫu tối thiểu ($)", 10, 10000, 200, 50)
        max_cand_a = col_a4.number_input("Số ví tối đa kiểm toán", 5, 100, 25, 5)

    elif seed_mode.startswith("Nguồn B"):
        col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
        lb_cat = col_b1.selectbox("Category Leaderboard", ["SPORTS", "FINANCE", "POLITICS", "OVERALL"])
        lb_period = col_b2.selectbox("Khung thời gian", ["MONTH", "ALL", "WEEK"])
        lb_limit = col_b3.number_input("Top N ví", 10, 50, 30, 5)

    else:
        raw_text = st.text_area(
            "Dán danh sách địa chỉ ví `0x...` (mỗi dòng 1 ví hoặc cách nhau dấu phẩy):",
            value=f"{BENCHMARK_CS2_WALLET}\n",
            height=110,
        )
        uploaded_file = st.file_uploader("Hoặc tải lên file CSV chứa cột địa chỉ ví (`0x...`)", type=["csv"])

        parsed_wallets = re.findall(r"0x[a-fA-F0-9]{40}", raw_text)
        if uploaded_file is not None:
            try:
                up_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                parsed_wallets.extend(re.findall(r"0x[a-fA-F0-9]{40}", up_content))
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")

        st.session_state["seeded_wallets"] = list(dict.fromkeys(w.lower() for w in parsed_wallets))

    include_benchmark = st.checkbox(
        f"Đính kèm ví chuẩn CS2 (`{BENCHMARK_CS2_WALLET[:10]}...`) vào bảng để làm mốc đối chiếu",
        value=True,
    )

    btn_col1, btn_col2 = st.columns([1, 2])
    seed_only_clicked = False
    if not seed_mode.startswith("Nguồn C"):
        seed_only_clicked = btn_col1.button("🌐 Chỉ thu thập ví hạt giống (Xem trước)", use_container_width=True)
    run_full_clicked = btn_col2.button(
        "🚀 ĐÀO VÍ & CHẠY KIỂM TOÁN NGAY",
        type="primary",
        use_container_width=True,
    )

    if seed_only_clicked or run_full_clicked:
        if seed_mode.startswith("Nguồn A"):
            with st.spinner("Đang quét Gamma API & Data API /v2/trades để thu thập danh sách ví..."):
                w_list, seed_df, logs = seed_wallets_from_markets(cat_choice, int(n_mkts), float(min_trd))
                st.session_state["seeded_wallets"] = w_list[: int(max_cand_a)]
                st.session_state["seed_preview"] = seed_df
                st.session_state["seed_logs"] = logs
        elif seed_mode.startswith("Nguồn B"):
            with st.spinner("Đang kéo danh sách ví từ Leaderboard Polymarket..."):
                w_list, seed_df, logs = seed_wallets_from_leaderboard(lb_cat, lb_period, int(lb_limit))
                st.session_state["seeded_wallets"] = w_list
                st.session_state["seed_preview"] = seed_df
                st.session_state["seed_logs"] = logs

    if st.session_state["seed_logs"]:
        for lg in st.session_state["seed_logs"]:
            st.caption(f"• {lg}")

    seeded_list = list(st.session_state.get("seeded_wallets", []))
    if include_benchmark and BENCHMARK_CS2_WALLET not in seeded_list:
        seeded_list = [BENCHMARK_CS2_WALLET] + seeded_list

    if "seed_preview" in st.session_state and not st.session_state["seed_preview"].empty and not seed_mode.startswith("Nguồn C"):
        with st.expander(f"📋 Xem danh sách {len(st.session_state['seed_preview'])} ví hạt giống vừa đào được", expanded=False):
            st.dataframe(st.session_state["seed_preview"], use_container_width=True, hide_index=True)

    if run_full_clicked and seeded_list:
        results = []
        details_map = dict(st.session_state["audit_details"])
        prog = st.progress(0.0, text="Bắt đầu kiểm toán...")

        for idx, w in enumerate(seeded_list):
            prog.progress((idx + 1) / len(seeded_list), text=f"Đang kiểm toán ví {idx + 1}/{len(seeded_list)}: `{w}`")
            try:
                summary, pos_df = evaluate_wallet_pro(w, cfg=cfg, max_pages=int(max_api_pages))
                results.append(summary)
                if not pos_df.empty:
                    details_map[w] = pos_df
            except Exception as e:
                results.append({
                    "wallet": w,
                    "verdict": "⚠️ LỖI API",
                    "score": 0,
                    "markets": 0,
                    "positions": 0,
                    "true_cost": 0.0,
                    "net_pnl": 0.0,
                    "true_roi": 0.0,
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
                    "red_flags": f"Lỗi tải dữ liệu: {e}",
                })

        prog.empty()
        res_df = pd.DataFrame(results).sort_values(["score", "cap_edge", "net_pnl"], ascending=[False, False, False]).reset_index(drop=True)
        st.session_state["audit_results"] = res_df
        st.session_state["audit_details"] = details_map

    res_df = st.session_state["audit_results"]
    if not res_df.empty:
        st.divider()
        pass_cnt = int((res_df["verdict"] == "✅ ĐẠT CHUẨN COPY").sum())
        watch_cnt = int((res_df["verdict"] == "🟡 THEO DÕI THÊM").sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tổng ví đã kiểm toán", f"{len(res_df)} ví")
        k2.metric("✅ Đạt chuẩn Copy-Trade", f"{pass_cnt} ví")
        k3.metric("🟡 Cần theo dõi thêm", f"{watch_cnt} ví")
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
                "true_cost": st.column_config.NumberColumn("Vốn mua 1 chiều", format="$%,.0f"),
                "net_pnl": st.column_config.NumberColumn("Net PnL (kể cả OPEN)", format="$%+,.0f"),
                "true_roi": st.column_config.NumberColumn("True ROI", format="%+.1f%%"),
                "cap_wr": st.column_config.NumberColumn("CapWR", format="%.1f%%"),
                "cap_edge": st.column_config.NumberColumn("CapEdge", format="%+.1f%%"),
                "z_score": st.column_config.NumberColumn("Z-score", format="%.2f"),
                "hold_pnl_share": st.column_config.NumberColumn("% Lãi REDEEM", format="%.0f%%"),
                "core_band_share": st.column_config.NumberColumn("% Vốn 20-85%", format="%.0f%%"),
                "pnl_ex_top5": st.column_config.NumberColumn("PnL ex-Top 5", format="$%+,.0f"),
                "med_size": st.column_config.NumberColumn("Size trung vị", format="$%.0f"),
                "med_accum_h": st.column_config.NumberColumn("Gom lệnh (h)", format="%.1fh"),
                "red_flags": st.column_config.TextColumn("Chẩn đoán & Cờ rủi ro", width="large"),
                "profile_url": st.column_config.LinkColumn("Polymarket", display_text="Mở Profile"),
            },
        )

        csv_out = res_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Tải bảng xếp hạng kiểm toán (CSV)",
            data=csv_out,
            file_name=f"polymarket_pro_wallets_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ==============================================================================
# TAB 2: DEEP SINGLE-WALLET X-RAY
# ==============================================================================
with tab_xray:
    audited_wallets = list(st.session_state["audit_details"].keys())
    col_x1, col_x2 = st.columns([2, 2])

    selected_from_list = col_x1.selectbox(
        "Chọn ví từ danh sách đã quét:",
        options=["-- Nhập thủ công bên phải --"] + audited_wallets,
    )
    manual_xray_wallet = col_x2.text_input(
        "Hoặc dán trực tiếp 1 địa chỉ ví `0x...` để soi sâu:",
        value="" if selected_from_list != "-- Nhập thủ công bên phải --" else BENCHMARK_CS2_WALLET,
    )

    target_xray = (
        selected_from_list
        if selected_from_list != "-- Nhập thủ công bên phải --"
        else manual_xray_wallet.strip().lower()
    )

    if st.button("🔬 Phân tích X-Ray ví này", use_container_width=True) and target_xray.startswith("0x"):
        with st.spinner(f"Đang giải phẫu toàn bộ sổ cái của `{target_xray}`..."):
            s_one, df_one = evaluate_wallet_pro(target_xray, cfg=cfg, max_pages=int(max_api_pages))
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
