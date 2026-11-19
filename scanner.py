import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

# ======================== CONFIG & CONSTANTS ========================
st.set_page_config(
    page_title="Polymarket Dual Signal Scanner (CS2 & Dota 2)",
    page_icon="⚡",
    layout="wide",
)

DEFAULT_CS2_WALLET = "0xc9bfc7986767173b4f1c3fc9b2c6b255fc7381c0"
DEFAULT_DOTA_INPUT = "@dotabotreborn"
DOTA_WALLET_PREFIX = "0x3231d2d67053c5dc3715f64291"

DATA_API_BASE = "https://data-api.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
STATE_FILE = "scanner_state.json"
VN_TZ = timezone(timedelta(hours=7))

EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

TEAM_ALIASES = {
    "vp": "virtus.pro",
    "baks": "baks esports",
    "nxs": "nexus",
    "ic.a": "inner circle academy",
    "ice": "inner circle esports",
    "ef": "eternal fire",
    "g1": "genone",
    "bca": "betclic apogee esports",
    "priv": "private",
    "ntr": "nuclear tigeres",
    "pha": "phantom",
    "all": "alliance",
    "bcg": "bc.game esports",
    "sng": "sangal",
    "g2.a": "g2 ares",
    "vit": "vitality",
    "prv": "parivision",
    "fal": "team falcons",
    "tl": "liquid",
    "mglz": "themongolz",
    "lg": "luminosity",
    "lgc": "legacy",
    "fnc": "fnatic",
    "hero": "heroic",
    "imp": "imperial",
    "aur": "aurora gaming",
    "bb": "betboom team",
    "ts": "spirit",
    "team spirit": "spirit",
    "navi": "natus vincere",
    "ast": "astralis",
    "gls": "galorys",
    "bar": "barca",
    "den": "dendele cs",
    "100t": "100 thieves",
    "gre": "grêmio esports",
    "tdp": "turma do pagode",
}


# ======================== PERSISTENCE HELPERS ========================
def short_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def load_persisted_ids(active_keys: set[str] | None = None) -> set[str]:
    saved: set[str] = set()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    saved.update(str(x) for x in data)
        except Exception:
            pass

    qp_raw = st.query_params.get("entered", "")
    if isinstance(qp_raw, list):
        qp_raw = ",".join(qp_raw)
    if qp_raw and active_keys:
        hashes = {h.strip() for h in str(qp_raw).split(",") if h.strip()}
        for k in active_keys:
            if short_hash(k) in hashes:
                saved.add(k)
    return saved


def persist_entered_ids(entered_ids: set[str], active_keys: set[str] | None = None) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(entered_ids), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        if entered_ids:
            target_keys = (entered_ids & active_keys) if active_keys else entered_ids
            hashes = sorted({short_hash(k) for k in target_keys})
            if hashes:
                st.query_params["entered"] = ",".join(hashes)
            elif "entered" in st.query_params:
                del st.query_params["entered"]
        else:
            if "entered" in st.query_params:
                del st.query_params["entered"]
    except Exception:
        pass


# ======================== WALLET RESOLVER & API FETCHERS ========================
def resolve_wallet_address(raw_input: str, expected_prefix: str = "") -> tuple[str, str]:
    """
    Chấp nhận cả địa chỉ 42 ký tự 0x... lẫn @username (ví dụ: @dotabotreborn)
    và tự động tra cứu địa chỉ proxyWallet đầy đủ qua Polymarket API.
    """
    cleaned = raw_input.strip()
    if not cleaned:
        return "", "Chưa nhập địa chỉ ví hoặc username."

    if EVM_ADDR_RE.match(cleaned):
        return cleaned.lower(), ""

    cache: dict[str, str] = st.session_state.setdefault("wallet_resolve_cache", {})
    cache_key = cleaned.lower()
    if cache_key in cache and EVM_ADDR_RE.match(cache[cache_key]):
        return cache[cache_key], ""

    handle = cleaned.lstrip("@").strip()
    prefix_l = expected_prefix.lower().strip()

    # 1. Tra cứu qua Gamma Public Search Profiles
    try:
        resp = requests.get(
            f"{GAMMA_API_BASE}/public-search",
            params={"q": handle, "search_profiles": "true", "limit_per_type": 10},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            profiles = data.get("profiles") or []
            for p in profiles:
                pw = str(p.get("proxyWallet") or p.get("address") or "").strip().lower()
                uname = str(p.get("name") or p.get("pseudonym") or "").strip().lower()
                if EVM_ADDR_RE.match(pw):
                    if (prefix_l and pw.startswith(prefix_l)) or (uname == handle.lower()):
                        cache[cache_key] = pw
                        return pw, ""
            for p in profiles:
                pw = str(p.get("proxyWallet") or p.get("address") or "").strip().lower()
                if EVM_ADDR_RE.match(pw):
                    cache[cache_key] = pw
                    return pw, ""
    except Exception:
        pass

    # 2. Fallback qua Data API Leaderboard (tìm theo userName)
    for ep in (f"{DATA_API_BASE}/v2/leaderboard", f"{DATA_API_BASE}/v1/leaderboard"):
        try:
            resp = requests.get(
                ep,
                params={"category": "OVERALL", "timePeriod": "ALL", "userName": handle, "limit": 10},
                timeout=10,
            )
            if resp.ok:
                payload = resp.json()
                rows = payload.get("data") or [] if isinstance(payload, dict) else (payload or [])
                for item in rows:
                    pw = str(
                        item.get("proxyWallet")
                        or item.get("proxy_wallet")
                        or item.get("user")
                        or ""
                    ).strip().lower()
                    if EVM_ADDR_RE.match(pw):
                        cache[cache_key] = pw
                        return pw, ""
        except Exception:
            pass

    # 3. Fallback đọc trang profile công khai polymarket.com/@username
    if not cleaned.lower().startswith("0x"):
        try:
            page = requests.get(
                f"https://polymarket.com/@{handle}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            if page.ok:
                if prefix_l:
                    m_pref = re.search(rf"({re.escape(prefix_l)}[a-f0-9]{{14}})", page.text, re.IGNORECASE)
                    if m_pref:
                        pw = m_pref.group(1).lower()
                        cache[cache_key] = pw
                        return pw, ""
                m_json = re.search(r'"proxyWallet"\s*:\s*"(0x[a-fA-F0-9]{40})"', page.text)
                if m_json:
                    pw = m_json.group(1).lower()
                    cache[cache_key] = pw
                    return pw, ""
        except Exception:
            pass

    return "", f"Không tự phân giải được '{cleaned}'. Hãy dán trực tiếp địa chỉ 42 ký tự (0x...) của ví vào ô cấu hình."


def fetch_positions_by_status(wallet: str, status: str) -> list[dict]:
    results: list[dict] = []
    cursor = None
    seen_cursors: set[str] = set()

    for _ in range(25):
        params = {
            "user": wallet.lower().strip(),
            "status": status,
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(f"{DATA_API_BASE}/v2/positions", params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        if isinstance(payload, list):
            results.extend(payload)
            break

        batch = payload.get("data") or payload.get("positions") or []
        results.extend(batch)

        next_cursor = payload.get("next_cursor") or payload.get("cursor")
        if not next_cursor or next_cursor in seen_cursors or not batch:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return results


def fetch_gamma_events_meta(
    condition_ids: list[str],
    token_ids: list[str],
    cache: dict[str, dict],
) -> dict[str, dict]:
    missing_cids = [c for c in dict.fromkeys(condition_ids) if c and c not in cache]
    for i in range(0, len(missing_cids), 40):
        chunk = missing_cids[i : i + 40]
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets",
                params={"condition_ids": ",".join(chunk), "limit": 100},
                timeout=12,
            )
            if resp.ok:
                markets = resp.json()
                if isinstance(markets, list):
                    for m in markets:
                        cid = str(m.get("conditionId") or m.get("condition_id") or "").strip()
                        events = m.get("events") or []
                        ev_title = ""
                        ev_slug = ""
                        if isinstance(events, list) and events:
                            ev_title = str(events[0].get("title") or "").strip()
                            ev_slug = str(events[0].get("slug") or "").strip()
                        if not ev_title:
                            ev_title = str(m.get("question") or "").strip()
                        if not ev_slug:
                            ev_slug = str(m.get("slug") or "").strip()

                        if cid and (ev_title or ev_slug):
                            cache[cid] = {
                                "event_title": ev_title,
                                "event_slug": ev_slug,
                                "question": str(m.get("question") or "").strip(),
                            }
        except Exception:
            pass

    still_missing_tokens = []
    token_to_cid: dict[str, str] = {}
    for cid, tid in zip(condition_ids, token_ids):
        if cid and cid not in cache and tid:
            still_missing_tokens.append(tid)
            token_to_cid[tid] = cid

    for i in range(0, len(still_missing_tokens), 40):
        chunk = still_missing_tokens[i : i + 40]
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets",
                params={"clob_token_ids": ",".join(chunk), "limit": 100},
                timeout=12,
            )
            if resp.ok:
                markets = resp.json()
                if isinstance(markets, list):
                    for m in markets:
                        cid = str(m.get("conditionId") or m.get("condition_id") or "").strip()
                        raw_tids = m.get("clobTokenIds") or []
                        if isinstance(raw_tids, str):
                            try:
                                raw_tids = json.loads(raw_tids)
                            except Exception:
                                raw_tids = []
                        if not cid:
                            for t in raw_tids:
                                if str(t) in token_to_cid:
                                    cid = token_to_cid[str(t)]
                                    break
                        events = m.get("events") or []
                        ev_title = str(events[0].get("title") or "").strip() if events else ""
                        ev_slug = str(events[0].get("slug") or "").strip() if events else ""
                        if not ev_title:
                            ev_title = str(m.get("question") or "").strip()
                        if not ev_slug:
                            ev_slug = str(m.get("slug") or "").strip()
                        if cid and (ev_title or ev_slug):
                            cache[cid] = {
                                "event_title": ev_title,
                                "event_slug": ev_slug,
                                "question": str(m.get("question") or "").strip(),
                            }
        except Exception:
            pass

    return cache


def fetch_clob_books(token_ids: list[str]) -> dict[str, dict]:
    books_map: dict[str, dict] = {}
    unique_tids = [t for t in dict.fromkeys(token_ids) if t]
    for i in range(0, len(unique_tids), 100):
        chunk = unique_tids[i : i + 100]
        payload = [{"token_id": tid} for tid in chunk]
        try:
            resp = requests.post(f"{CLOB_API_BASE}/books", json=payload, timeout=15)
            if resp.ok:
                data = resp.json()
                if isinstance(data, list):
                    for book in data:
                        aid = str(book.get("asset_id") or book.get("token_id") or "").strip()
                        if aid:
                            books_map[aid] = book
        except Exception:
            pass
    return books_map


# ======================== PARSING & FILTER LOGIC ========================
def safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return default


def parse_time_info(raw_ts) -> tuple[str, float | None]:
    """Trả về (chuỗi hiển thị thời gian trước, số phút tuổi lệnh)."""
    if not raw_ts:
        return "—", None
    try:
        if isinstance(raw_ts, (int, float)):
            ts = float(raw_ts)
            if ts > 1e12:
                ts /= 1000.0
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            s = str(raw_ts).strip()
            if s.isdigit():
                ts = float(s)
                if ts > 1e12:
                    ts /= 1000.0
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
        diff = datetime.now(timezone.utc) - dt
        age_mins = max(0.0, diff.total_seconds() / 60.0)
        mins = int(age_mins)
        if mins < 1:
            return "Vừa xong ⚡", round(age_mins, 1)
        if mins < 60:
            return f"{mins}p trước", round(age_mins, 1)
        hours = mins // 60
        rem_mins = mins % 60
        if hours < 24:
            return f"{hours}h{rem_mins:02d}p trước", round(age_mins, 1)
        days = hours // 24
        return f"{days}d {hours % 24}h trước", round(age_mins, 1)
    except Exception:
        return "—", None


def normalize_team_name(name: str) -> str:
    clean_name = re.sub(r"\s+", " ", name).strip().lower()
    return TEAM_ALIASES.get(clean_name, clean_name)


def extract_team_pair(title: str) -> frozenset[str] | None:
    clean = re.sub(
        r"^(Counter-Strike|CS2|Dota\s*2|Map\s*\d+\s*Rounds\s*Handicap|Map\s*\d+\s*Winner|Game\s*\d+\s*Winner|Map\s*Handicap|Total\s*Maps|Match\s*Winner|Winner)\s*:\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\([^)]*\)", "", clean)
    clean = re.split(r"\s+-\s+|\s+\|\s+", clean)[0].strip()
    m = re.split(r"\s+vs\.?\s+", clean, maxsplit=1, flags=re.IGNORECASE)
    if len(m) == 2:
        t1 = normalize_team_name(m[0])
        t2 = normalize_team_name(m[1])
        if len(t1) >= 2 and len(t2) >= 2:
            return frozenset({t1, t2})
    return None


def matches_blocked_league(
    combined_text: str,
    blocked_keywords: list[str],
) -> tuple[bool, str]:
    if not blocked_keywords:
        return False, ""
    txt = combined_text.lower()
    is_cct_sa = "cct south america" in txt or "cct sa" in txt

    for kw in blocked_keywords:
        if kw == "cct europe":
            if not is_cct_sa and (
                re.search(r"\bcct\b.{0,28}\beurope\b", txt)
                or re.search(r"\bcct[\s\-_]*eu\b", txt)
                or "cct europe" in txt
            ):
                return True, "CCT Europe"
        elif kw and kw in txt:
            return True, kw
    return False, ""


def prepare_wallet_candidates(
    profile_cfg: dict,
    open_positions: list[dict],
    gamma_cache: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """
    Tiền xử lý danh sách vị thế OPEN của 1 profile ví (CS2 hoặc Dota 2),
    trả về (danh sách dòng đã phân tích, danh sách token_id cần tra sổ lệnh CLOB).
    """
    source_tag = profile_cfg["source_tag"]
    wallet = profile_cfg["wallet"]
    multiplier = float(profile_cfg["multiplier"])
    min_user_bet = float(profile_cfg["min_user_bet"])
    min_prob_pct = int(profile_cfg["min_prob_pct"])
    max_fav_prob = int(profile_cfg["max_fav_prob"])
    block_50_60 = bool(profile_cfg["block_50_60"])
    detect_hedge = bool(profile_cfg["detect_hedge"])
    blocked_keywords = profile_cfg["blocked_keywords"]

    parsed_rows: list[dict] = []
    blocked_team_pairs: set[frozenset[str]] = set()
    seen_pos_keys: set[str] = set()

    for p in open_positions:
        cid = str(p.get("condition_id") or p.get("conditionId") or "").strip()
        tid = str(p.get("token_id") or p.get("asset") or "").strip()
        title = str(p.get("title") or p.get("question") or "Unknown Market").strip()
        outcome = str(p.get("outcome") or p.get("side") or "").strip()
        slug = str(p.get("slug") or p.get("event_slug") or "").strip()

        pos_key = f"{wallet[:10]}_{cid}_{tid}_{outcome}"
        if pos_key in seen_pos_keys:
            continue
        seen_pos_keys.add(pos_key)

        meta = gamma_cache.get(cid, {})
        ev_title = meta.get("event_title", "")
        ev_slug = meta.get("event_slug", "")
        if not slug and ev_slug:
            slug = ev_slug
        gamma_unverified = not bool(ev_title or ev_slug)

        avg_p = safe_float(p.get("avg_price") or p.get("avgPrice"), 0.0)
        cur_p = safe_float(p.get("current_price") or p.get("curPrice"), 0.0)
        cur_size = safe_float(p.get("current_size") or p.get("size"), 0.0)
        cur_val = safe_float(p.get("current_value") or p.get("currentValue"), 0.0)
        if cur_p <= 0 and cur_size > 0 and cur_val > 0:
            cur_p = cur_val / cur_size

        entry_cost = safe_float(
            p.get("entry_cost_usdc") or p.get("initial_value") or p.get("initialValue"),
            0.0,
        )
        if entry_cost <= 0 and cur_size > 0 and avg_p > 0:
            entry_cost = cur_size * avg_p

        realized_pnl = safe_float(p.get("realized_pnl") or p.get("realizedPnl"), 0.0)
        unrealized_pnl = safe_float(p.get("unrealized_pnl") or p.get("unrealizedPnl"), 0.0)
        if unrealized_pnl == 0.0 and cur_val > 0 and entry_cost > 0:
            unrealized_pnl = cur_val - entry_cost
        total_pnl = realized_pnl + unrealized_pnl

        last_ts = (
            p.get("last_event_at")
            or p.get("updated_at")
            or p.get("created_at")
            or p.get("timestamp")
        )
        last_ago_str, age_mins = parse_time_info(last_ts)
        url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"

        combined_text = f"{title} | {ev_title} | {slug} | {ev_slug}"
        is_blk, blk_reason = matches_blocked_league(combined_text, blocked_keywords)
        team_pair = extract_team_pair(title)
        if is_blk and team_pair:
            blocked_team_pairs.add(team_pair)

        parsed_rows.append(
            {
                "source_tag": source_tag,
                "profile_cfg": profile_cfg,
                "pos_key": pos_key,
                "condition_id": cid,
                "token_id": tid,
                "title": title,
                "event_title": ev_title,
                "outcome": outcome,
                "slug": slug,
                "url": url,
                "avg_price": avg_p,
                "cur_price": cur_p,
                "entry_cost_usdc": entry_cost,
                "total_pnl": total_pnl,
                "last_buy_ago": last_ago_str,
                "age_mins": age_mins,
                "gamma_unverified": gamma_unverified,
                "is_blocked_direct": is_blk,
                "blocked_reason": blk_reason,
                "team_pair": team_pair,
            }
        )

    # Layer 3: Cross-lock secondary markets matching blocked league team pairs
    for row in parsed_rows:
        if not row["is_blocked_direct"] and row["team_pair"] and row["team_pair"] in blocked_team_pairs:
            row["is_blocked_direct"] = True
            row["blocked_reason"] = "Khóa chéo cặp đội giải chặn"

    cid_outcomes: dict[str, set[str]] = {}
    for row in parsed_rows:
        if 0.02 < row["cur_price"] < 0.98 and not row["is_blocked_direct"]:
            cid_outcomes.setdefault(row["condition_id"], set()).add(row["outcome"])

    candidate_tids: list[str] = []
    for row in parsed_rows:
        row["cid_outcome_cnt"] = len(cid_outcomes.get(row["condition_id"], set()))
        if not (0.02 < row["cur_price"] < 0.98):
            continue
        if row["is_blocked_direct"]:
            continue
        if detect_hedge and row["cid_outcome_cnt"] >= 2:
            continue
        raw_bet = row["entry_cost_usdc"] * multiplier
        if raw_bet < min_user_bet:
            continue
        avg_p = row["avg_price"]
        if avg_p <= 0.0 or avg_p < (min_prob_pct / 100.0) or avg_p > (max_fav_prob / 100.0):
            continue
        if block_50_60 and (0.50 <= avg_p <= 0.60):
            continue
        candidate_tids.append(row["token_id"])

    return parsed_rows, candidate_tids


def evaluate_all_profiles(profiles_data: list[tuple[dict, list[dict], list[dict]]]) -> dict:
    """
    Chạy đánh giá hợp nhất cho tất cả các profile ví đang bật (CS2 + Dota 2).
    """
    if "gamma_meta_cache" not in st.session_state:
        st.session_state["gamma_meta_cache"] = {}

    all_cids: list[str] = []
    all_tids: list[str] = []
    for _, open_pos, _ in profiles_data:
        for p in open_pos:
            all_cids.append(str(p.get("condition_id") or p.get("conditionId") or "").strip())
            all_tids.append(str(p.get("token_id") or p.get("asset") or "").strip())

    gamma_cache = fetch_gamma_events_meta(all_cids, all_tids, st.session_state["gamma_meta_cache"])

    all_parsed_rows: list[dict] = []
    all_candidate_tids: list[str] = []
    total_redeemable = 0

    for p_cfg, open_pos, redeem_pos in profiles_data:
        total_redeemable += len(redeem_pos)
        rows, cand_tids = prepare_wallet_candidates(p_cfg, open_pos, gamma_cache)
        all_parsed_rows.extend(rows)
        all_candidate_tids.extend(cand_tids)

    books_map = fetch_clob_books(all_candidate_tids)

    valid_list: list[dict] = []
    slip_list: list[dict] = []
    hedge_list: list[dict] = []
    blocked_list: list[dict] = []
    rejected_list: list[dict] = []
    dead_zero_count = 0

    for row in all_parsed_rows:
        p_cfg = row["profile_cfg"]
        multiplier = float(p_cfg["multiplier"])
        min_user_bet = float(p_cfg["min_user_bet"])
        max_user_cap = float(p_cfg["max_user_cap"])
        max_slippage_cents = float(p_cfg["max_slippage_cents"])
        drop_warn_cents = float(p_cfg["drop_warn_cents"])
        min_prob_pct = int(p_cfg["min_prob_pct"])
        max_fav_prob = int(p_cfg["max_fav_prob"])
        block_50_60 = bool(p_cfg["block_50_60"])
        detect_hedge = bool(p_cfg["detect_hedge"])
        max_signal_age_mins = int(p_cfg.get("max_signal_age_mins", 0))

        cur_p = row["cur_price"]
        avg_p = row["avg_price"]
        entry_cost = row["entry_cost_usdc"]
        raw_bet = entry_cost * multiplier
        final_bet = round(min(raw_bet, max_user_cap), 2)

        # 1. Vị thế đã ngã ngũ (0¢ hoặc 100¢) chờ Redeem
        if cur_p >= 0.98 or cur_p <= 0.02:
            if cur_p <= 0.02:
                dead_zero_count += 1
            row["reject_reason"] = f"Đã ngã ngũ (giá {cur_p * 100:.1f}¢)"
            rejected_list.append(row)
            continue

        # 2. Giải chặn (CCT Europe cho CS2)
        if row["is_blocked_direct"]:
            cond_notes = []
            if raw_bet < min_user_bet:
                cond_notes.append(f"< Min ${min_user_bet:.2f}")
            if block_50_60 and (0.50 <= avg_p <= 0.60):
                cond_notes.append("Dải 50-60%")
            elif avg_p > (max_fav_prob / 100.0) or avg_p < (min_prob_pct / 100.0):
                cond_notes.append(f"Ngoài dải {min_prob_pct}-{max_fav_prob}%")
            row["raw_user_bet"] = round(raw_bet, 2)
            row["other_cond"] = (
                f"Đạt vốn & giá ({row['blocked_reason']})"
                if not cond_notes
                else f"Thêm lỗi: {', '.join(cond_notes)}"
            )
            blocked_list.append(row)
            continue

        # 3. Hedge 2 đầu trên cùng condition_id
        if detect_hedge and row["cid_outcome_cnt"] >= 2:
            row["raw_user_bet"] = round(raw_bet, 2)
            row["hedge_note"] = "Ví gốc đang mở >= 2 cửa cùng trận"
            hedge_list.append(row)
            continue

        # 4. Ngưỡng vốn tối thiểu
        if raw_bet < min_user_bet:
            min_whale_req = min_user_bet / multiplier if multiplier > 0 else 0
            row["reject_reason"] = (
                f"Vốn ví ${entry_cost:,.0f} (< ${min_whale_req:,.0f}) -> ${raw_bet:.2f} < Min ${min_user_bet:.2f}"
            )
            rejected_list.append(row)
            continue

        # 5. Bộ lọc dải xác suất theo từng chiến lược
        if avg_p <= 0.0 or avg_p < (min_prob_pct / 100.0):
            row["reject_reason"] = f"Kèo Underdog sâu < {min_prob_pct}% ({avg_p * 100:.1f}¢)"
            rejected_list.append(row)
            continue
        if avg_p > (max_fav_prob / 100.0):
            row["reject_reason"] = f"Kèo chắc ăn > {max_fav_prob}% ({avg_p * 100:.1f}¢)"
            rejected_list.append(row)
            continue
        if block_50_60 and (0.50 <= avg_p <= 0.60):
            row["reject_reason"] = f"Vùng cấm 50%–60% ({avg_p * 100:.1f}¢)"
            rejected_list.append(row)
            continue

        # 6. Đánh giá sổ lệnh CLOB & Tốc độ Taker
        max_limit_price = min(avg_p + (max_slippage_cents / 100.0), 0.99)
        max_limit_cents = round(max_limit_price * 100.0, 1)

        book = books_map.get(row["token_id"], {})
        asks_raw = book.get("asks") or []
        parsed_asks: list[tuple[float, float]] = []
        for a in asks_raw:
            ap = safe_float(a.get("price"), 0.0)
            asz = safe_float(a.get("size"), 0.0)
            if ap > 0 and asz > 0:
                parsed_asks.append((ap, asz))
        parsed_asks.sort(key=lambda x: x[0])

        notes: list[str] = []
        age_mins = row.get("age_mins")
        if max_signal_age_mins > 0 and age_mins is not None:
            if age_mins <= 5.0:
                notes.append(f"⚡ Mới {age_mins:.0f}p")
            elif age_mins > max_signal_age_mins:
                notes.append(f"⏱️ Trễ {age_mins:.0f}p — CHECK IN-GAME!")

        if row["gamma_unverified"]:
            notes.append("⚠️ Chưa xác minh giải (Gamma)")

        if parsed_asks:
            best_ask = parsed_asks[0][0]
            avail_liq_usdc = sum(
                ap * asz for ap, asz in parsed_asks if ap <= max_limit_price + 1e-6
            )
            exec_price = best_ask
            is_empty_ask = False
        else:
            avail_liq_usdc = 0.0
            exec_price = cur_p if (0.02 < cur_p < 0.98) else avg_p
            is_empty_ask = True
            notes.append("📋 Trống Ask (Kê Limit)")

        slippage_cents = round((exec_price - avg_p) * 100.0, 1)
        drop_cents = round((avg_p - exec_price) * 100.0, 1)

        if drop_cents > drop_warn_cents:
            notes.append(f"⚠️ Tụt >{drop_warn_cents:.0f}¢ — CHECK LIVE!")

        is_slippage_over = slippage_cents > max_slippage_cents
        if not is_empty_ask and not is_slippage_over and avail_liq_usdc < final_bet:
            notes.append(f"💧 Ask mỏng (${avail_liq_usdc:.2f})")

        ref_limit_price = max(max_limit_price, 0.01)
        est_shares = final_bet / ref_limit_price
        needs_limit = is_empty_ask or is_slippage_over or (avail_liq_usdc < final_bet)
        if needs_limit and est_shares < 5.0:
            min_limit_usdc = 5.0 * ref_limit_price
            notes.append(f"⚠️ Limit cần >=5sh (~${min_limit_usdc:.2f})")

        if not notes:
            notes.append("✅ Khớp đẹp")

        row["final_user_bet"] = final_bet
        row["exec_price"] = exec_price
        row["slippage_cents"] = slippage_cents
        row["max_limit_cents"] = max_limit_cents
        row["avail_liq_usdc"] = round(avail_liq_usdc, 2)
        row["action_note"] = " | ".join(notes)

        if is_slippage_over:
            slip_list.append(row)
        else:
            valid_list.append(row)

    return {
        "valid": pd.DataFrame(valid_list),
        "slip": pd.DataFrame(slip_list),
        "hedge": pd.DataFrame(hedge_list),
        "blocked": pd.DataFrame(blocked_list),
        "rejected": pd.DataFrame(rejected_list),
        "redeemable_count": total_redeemable,
        "dead_zero_count": dead_zero_count,
    }


# ======================== SESSION STATE INIT ========================
if "entered_ids" not in st.session_state:
    st.session_state["entered_ids"] = load_persisted_ids()
if "editor_ver" not in st.session_state:
    st.session_state["editor_ver"] = 0
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None
if "last_scan_time" not in st.session_state:
    st.session_state["last_scan_time"] = None


# ======================== SIDEBAR CONFIG (2 PROFILES) ========================
with st.sidebar:
    st.header("🎛️ Cấu Hình 2 Chiến Lược Copy")

    # --- PROFILE 1: CS2 ---
    with st.expander("🎮 Ví 1: CS2 Whale (Pre-Match)", expanded=True):
        enable_cs2 = st.checkbox("Bật quét ví CS2", value=True, key="en_cs2")
        cs2_wallet_raw = st.text_input(
            "Địa chỉ ví CS2",
            value=DEFAULT_CS2_WALLET,
            key="w_cs2",
        ).strip()

        cs2_mult = st.number_input(
            "Hệ số nhân CS2",
            min_value=0.00001,
            max_value=1.0,
            value=0.00100,
            step=0.00010,
            format="%.5f",
            key="m_cs2",
        )
        c1_a, c1_b = st.columns(2)
        cs2_min_bet = c1_a.number_input(
            "Min lệnh ($)",
            min_value=0.1,
            value=2.00,
            step=0.5,
            key="min_cs2",
            help="0.00100 × $2,000 = $2.00",
        )
        cs2_max_cap = c1_b.number_input(
            "Trần tối đa ($)",
            min_value=1.0,
            value=20.00,
            step=1.0,
            key="max_cs2",
        )
        cs2_slip = st.slider(
            "Trượt giá Ask tối đa (¢)",
            0.5,
            10.0,
            2.5,
            0.5,
            key="slip_cs2",
        )
        cs2_drop_warn = st.slider(
            "Cảnh báo giá tụt sâu (¢)",
            2.0,
            15.0,
            5.0,
            1.0,
            key="drop_cs2",
        )
        cs2_block_50_60 = st.checkbox("Chặn dải 50% – 60%", value=True, key="b5060_cs2")
        cs2_max_fav = st.slider("Chặn kèo chắc ăn trên (%)", 65, 99, 85, 5, key="fav_cs2")
        cs2_blocked_kw = st.text_input(
            "Giải chặn (CS2)",
            value="CCT Europe",
            key="blk_cs2",
        )
        cs2_min_whale = cs2_min_bet / cs2_mult if cs2_mult > 0 else 0
        cs2_max_whale = cs2_max_cap / cs2_mult if cs2_mult > 0 else 0
        st.caption(f"Ngưỡng ví CS2: **${cs2_min_whale:,.0f} – ${cs2_max_whale:,.0f}**")

    # --- PROFILE 2: DOTA 2 ---
    with st.expander("🐉 Ví 2: DotaBotReborn (Post-Draft Taker)", expanded=True):
        enable_dota = st.checkbox("Bật quét ví Dota 2", value=True, key="en_dota")
        dota_wallet_raw = st.text_input(
            "Username hoặc Địa chỉ ví Dota",
            value=DEFAULT_DOTA_INPUT,
            key="w_dota",
            help="Nhập @dotabotreborn (tool tự tra địa chỉ 0x...) hoặc dán trực tiếp địa chỉ 42 ký tự 0x...",
        ).strip()

        dota_mult = st.number_input(
            "Hệ số nhân Dota",
            min_value=0.00001,
            max_value=1.0,
            value=0.00400,
            step=0.00050,
            format="%.5f",
            key="m_dota",
            help="Với hệ số 0.00400: lệnh $250 của DotaBotReborn -> bạn vào $1.00; lệnh $4,000 -> chạm trần $16.00.",
        )
        c2_a, c2_b = st.columns(2)
        dota_min_bet = c2_a.number_input(
            "Min lệnh ($)",
            min_value=0.1,
            value=1.00,
            step=0.5,
            key="min_dota",
            help="0.00400 × $250 = $1.00 (loại toàn bộ lệnh thăm dò < $250).",
        )
        dota_max_cap = c2_b.number_input(
            "Trần tối đa ($)",
            min_value=1.0,
            value=16.00,
            step=1.0,
            key="max_dota",
        )
        dota_slip = st.slider(
            "Trượt giá Ask tối đa (¢)",
            0.5,
            10.0,
            2.5,
            0.5,
            key="slip_dota",
        )
        dota_age_warn = st.slider(
            "Cảnh báo trễ lệnh Taker sau (phút)",
            3,
            30,
            10,
            1,
            key="age_dota",
            help="DotaBotReborn đánh 1 nhịp trước giờ đấu; nếu trễ quá số phút này cần kiểm tra trận đã vào combat chưa.",
        )
        c2_p1, c2_p2 = st.columns(2)
        dota_min_prob = c2_p1.number_input("Dải giá từ (%)", 5, 60, 35, 5, key="minp_dota")
        dota_max_fav = c2_p2.number_input("Đến tối đa (%)", 50, 95, 70, 5, key="maxp_dota")
        dota_block_50_60 = st.checkbox(
            "Chặn dải 50% – 60% (Dota)",
            value=False,
            key="b5060_dota",
            help="Tắt mặc định vì vùng 50-60% là vùng lãi cao nhất của DotaBotReborn (+35.7% ROI).",
        )
        dota_min_whale = dota_min_bet / dota_mult if dota_mult > 0 else 0
        dota_max_whale = dota_max_cap / dota_mult if dota_mult > 0 else 0
        st.caption(f"Ngưỡng ví Dota: **${dota_min_whale:,.0f} – ${dota_max_whale:,.0f}** | Dải **{dota_min_prob}%–{dota_max_fav}%**")

    detect_hedge_global = st.checkbox("Chặn kèo ví gốc Hedge 2 đầu (Chung)", value=True)

    st.divider()
    if st.session_state["entered_ids"]:
        if st.button(
            f"🧹 Xóa dấu Đã vào ({len(st.session_state['entered_ids'])})",
            use_container_width=True,
        ):
            st.session_state["entered_ids"] = set()
            st.session_state["editor_ver"] += 1
            persist_entered_ids(set(), active_keys=set())
            st.rerun()


# ======================== MAIN UI ========================
st.title("⚡ Trạm Quét Lệnh Copy-Trade Đa Chiến Lược (CS2 + Dota 2)")

col_btn, col_time = st.columns([2, 3])
scan_clicked = col_btn.button("🚀 QUÉT LỆNH HỢP LỆ NGAY", type="primary", use_container_width=True)

if scan_clicked or st.session_state["scan_results"] is None:
    with st.spinner("Đang phân giải địa chỉ ví, kéo Polymarket Data API v2, Gamma & Sổ lệnh CLOB..."):
        active_profiles: list[dict] = []
        resolve_warnings: list[str] = []

        if enable_cs2 and cs2_wallet_raw:
            cs2_addr, cs2_err = resolve_wallet_address(cs2_wallet_raw)
            if cs2_addr:
                active_profiles.append(
                    {
                        "source_tag": "🎮 CS2",
                        "wallet": cs2_addr,
                        "multiplier": cs2_mult,
                        "min_user_bet": cs2_min_bet,
                        "max_user_cap": cs2_max_cap,
                        "max_slippage_cents": cs2_slip,
                        "drop_warn_cents": cs2_drop_warn,
                        "min_prob_pct": 0,
                        "max_fav_prob": cs2_max_fav,
                        "block_50_60": cs2_block_50_60,
                        "detect_hedge": detect_hedge_global,
                        "blocked_keywords": [k.strip().lower() for k in cs2_blocked_kw.split(",") if k.strip()],
                        "max_signal_age_mins": 0,
                    }
                )
            elif cs2_err:
                resolve_warnings.append(f"[CS2] {cs2_err}")

        if enable_dota and dota_wallet_raw:
            dota_addr, dota_err = resolve_wallet_address(
                dota_wallet_raw,
                expected_prefix=DOTA_WALLET_PREFIX if "dotabotreborn" in dota_wallet_raw.lower() else "",
            )
            if dota_addr:
                active_profiles.append(
                    {
                        "source_tag": "🐉 Dota 2",
                        "wallet": dota_addr,
                        "multiplier": dota_mult,
                        "min_user_bet": dota_min_bet,
                        "max_user_cap": dota_max_cap,
                        "max_slippage_cents": dota_slip,
                        "drop_warn_cents": 5.0,
                        "min_prob_pct": dota_min_prob,
                        "max_fav_prob": dota_max_fav,
                        "block_50_60": dota_block_50_60,
                        "detect_hedge": detect_hedge_global,
                        "blocked_keywords": [],
                        "max_signal_age_mins": dota_age_warn,
                    }
                )
            elif dota_err:
                resolve_warnings.append(f"[Dota 2] {dota_err}")

        for w_msg in resolve_warnings:
            st.warning(w_msg)

        if not active_profiles:
            st.error("Chưa có ví hợp lệ nào được bật để quét.")
        else:
            try:
                profiles_payload = []
                for p_cfg in active_profiles:
                    open_pos = fetch_positions_by_status(p_cfg["wallet"], "OPEN")
                    redeem_pos = fetch_positions_by_status(p_cfg["wallet"], "REDEEMABLE")
                    profiles_payload.append((p_cfg, open_pos, redeem_pos))

                res = evaluate_all_profiles(profiles_payload)
                res["resolved_info"] = {p["source_tag"]: p["wallet"] for p in active_profiles}
                st.session_state["scan_results"] = res
                st.session_state["last_scan_time"] = datetime.now(VN_TZ).strftime("%H:%M:%S %d/%m/%Y (GMT+7)")

                active_keys: set[str] = set()
                for k_df in ("valid", "slip"):
                    if not res[k_df].empty:
                        active_keys.update(res[k_df]["pos_key"].tolist())
                st.session_state["entered_ids"] = load_persisted_ids(active_keys)
            except Exception as exc:
                st.error(f"Lỗi kết nối Polymarket API: {exc}")

if st.session_state["last_scan_time"]:
    res_meta = (st.session_state["scan_results"] or {}).get("resolved_info", {})
    wallets_badge = " | ".join(f"{tag}: `{addr[:8]}...{addr[-4:]}`" for tag, addr in res_meta.items())
    col_time.info(f"🕒 Cập nhật: **{st.session_state['last_scan_time']}** • {wallets_badge}")

with st.expander("📋 So Sánh Nhanh Bộ Quy Tắc 2 Ví Đang Chạy Song Song (Bấm để xem)", expanded=False):
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown(
            "**🎮 Chiến lược 1: CS2 Whale (Pre-Match)**\n"
            f"* **Ngưỡng vốn:** Ví gốc `>= ${cs2_min_whale:,.0f}` (bạn vào `${cs2_min_bet:.2f} – ${cs2_max_cap:.2f}`).\n"
            f"* **Dải giá & Giải:** Lấy `< 50%` và `60% – {cs2_max_fav}%`; chặn `50%–60%`, Hedge 2 đầu & **{cs2_blocked_kw}**.\n"
            f"* **Đặc thù:** Gom nhiều lệnh trước giờ đấu, chỉ cần kiểm tra trận chưa đá Live."
        )
    with r_col2:
        st.markdown(
            "**🐉 Chiến lược 2: DotaBotReborn (Post-Draft Taker)**\n"
            f"* **Ngưỡng vốn:** Ví gốc `>= ${dota_min_whale:,.0f}` (bạn vào `${dota_min_bet:.2f} – ${dota_max_cap:.2f}`).\n"
            f"* **Dải giá:** Lấy **`{dota_min_prob}% – {dota_max_fav}%`** (mở khóa `50%–60%`, chặn `> {dota_max_fav}%` và `< {dota_min_prob}%`).\n"
            f"* **Đặc thù:** Đánh 1 nhịp Taker ngay trước giờ game bắt đầu -> ưu tiên lệnh mới `<= {dota_age_warn} phút` và trượt giá `<= +{dota_slip:.1f}¢`."
        )

res = st.session_state["scan_results"]
if res is not None:
    valid_df: pd.DataFrame = res["valid"].copy()
    slip_df: pd.DataFrame = res["slip"].copy()
    hedge_df: pd.DataFrame = res["hedge"].copy()
    blocked_df: pd.DataFrame = res["blocked"].copy()
    rejected_df: pd.DataFrame = res["rejected"].copy()
    redeem_cnt: int = res["redeemable_count"]
    dead_zero_cnt: int = res.get("dead_zero_count", 0)

    entered_set: set[str] = st.session_state["entered_ids"]
    active_keys_all: set[str] = set()
    if not valid_df.empty:
        active_keys_all.update(valid_df["pos_key"].tolist())
    if not slip_df.empty:
        active_keys_all.update(slip_df["pos_key"].tolist())

    remaining_capital = 0.0
    if not valid_df.empty:
        remaining_capital = float(
            valid_df.loc[~valid_df["pos_key"].isin(entered_set), "final_user_bet"].sum()
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🟢 Kèo hợp lệ ngay", len(valid_df))
    m2.metric("💰 Vốn cần thêm ($)", f"${remaining_capital:.2f}")
    m3.metric("🟡 Quá trượt giá (Limit)", len(slip_df))
    m4.metric("🚫 Giải chặn (CCT EU)", len(blocked_df))
    m5.metric("🏆 Chờ Redeem / Xác 0¢", f"{redeem_cnt} / {dead_zero_cnt}")

    st.divider()

    # --- MỤC 1: KÈO HỢP LỆ ĐỂ VÀO LỆNH NGAY ---
    st.subheader(f"🟢 1. Kèo Hợp Lệ Để Vào Lệnh Ngay ({len(valid_df)})")
    if valid_df.empty:
        st.info("Hiện chưa có kèo nào từ CS2 hoặc Dota 2 khớp đủ quy tắc và nằm trong ngưỡng trượt giá Ask.")
    else:
        valid_df["entered"] = valid_df["pos_key"].isin(entered_set)
        disp_valid = valid_df[
            [
                "pos_key",
                "entered",
                "source_tag",
                "title",
                "outcome",
                "final_user_bet",
                "action_note",
                "avg_price",
                "exec_price",
                "slippage_cents",
                "max_limit_cents",
                "avail_liq_usdc",
                "entry_cost_usdc",
                "last_buy_ago",
                "url",
            ]
        ].copy()
        disp_valid["avg_price"] = (disp_valid["avg_price"] * 100.0).round(1)
        disp_valid["exec_price"] = (disp_valid["exec_price"] * 100.0).round(1)

        edited_valid = st.data_editor(
            disp_valid,
            key=f"valid_editor_{st.session_state['editor_ver']}",
            use_container_width=True,
            hide_index=True,
            column_order=[
                "entered",
                "source_tag",
                "title",
                "outcome",
                "final_user_bet",
                "action_note",
                "avg_price",
                "exec_price",
                "slippage_cents",
                "max_limit_cents",
                "avail_liq_usdc",
                "entry_cost_usdc",
                "last_buy_ago",
                "url",
            ],
            column_config={
                "entered": st.column_config.CheckboxColumn("✅ Đã vào", default=False),
                "source_tag": st.column_config.TextColumn("Nguồn Ví", width="small"),
                "title": st.column_config.TextColumn("Trận đấu / Kèo", width="medium"),
                "outcome": st.column_config.TextColumn("Cửa chọn", width="small"),
                "final_user_bet": st.column_config.NumberColumn("Tiền bạn vào ($)", format="$%.2f"),
                "action_note": st.column_config.TextColumn("Trạng thái sổ lệnh", width="medium"),
                "avg_price": st.column_config.NumberColumn("Giá ví gốc (¢)", format="%.1f¢"),
                "exec_price": st.column_config.NumberColumn("Best Ask (¢)", format="%.1f¢"),
                "slippage_cents": st.column_config.NumberColumn("Trượt giá (¢)", format="%+.1f¢"),
                "max_limit_cents": st.column_config.NumberColumn("🛑 Giá trần (¢)", format="%.1f¢"),
                "avail_liq_usdc": st.column_config.NumberColumn("Thanh khoản Ask ($)", format="$%.2f"),
                "entry_cost_usdc": st.column_config.NumberColumn("Vốn ví gốc ($)", format="$%,.0f"),
                "last_buy_ago": st.column_config.TextColumn("Cập nhật", width="small"),
                "url": st.column_config.LinkColumn("Polymarket", display_text="Mở kèo ↗"),
            },
            disabled=[
                "pos_key",
                "source_tag",
                "title",
                "outcome",
                "final_user_bet",
                "action_note",
                "avg_price",
                "exec_price",
                "slippage_cents",
                "max_limit_cents",
                "avail_liq_usdc",
                "entry_cost_usdc",
                "last_buy_ago",
                "url",
            ],
        )

        new_entered_valid = set(edited_valid.loc[edited_valid["entered"], "pos_key"].tolist())
        old_entered_valid = set(disp_valid.loc[disp_valid["entered"], "pos_key"].tolist())
        if new_entered_valid != old_entered_valid:
            updated_set = (entered_set - old_entered_valid) | new_entered_valid
            st.session_state["entered_ids"] = updated_set
            persist_entered_ids(updated_set, active_keys_all)
            st.rerun()

    # --- MỤC 2: KÈO ĐẠT ĐIỀU KIỆN NHƯNG QUÁ TRƯỢT GIÁ ASK ---
    if not slip_df.empty:
        st.subheader(f"🟡 2. Kèo Đạt Điều Kiện Nhưng Quá Trượt Giá Ask ({len(slip_df)})")
        st.caption(
            "Giá Ask trên sổ lệnh đang cao hơn trần cho phép; chỉ đặt lệnh Limit ở cột **🛑 Giá trần** nếu trận chưa bắt đầu."
        )
        slip_df["entered"] = slip_df["pos_key"].isin(st.session_state["entered_ids"])
        disp_slip = slip_df[
            [
                "pos_key",
                "entered",
                "source_tag",
                "title",
                "outcome",
                "final_user_bet",
                "action_note",
                "avg_price",
                "exec_price",
                "slippage_cents",
                "max_limit_cents",
                "entry_cost_usdc",
                "last_buy_ago",
                "url",
            ]
        ].copy()
        disp_slip["avg_price"] = (disp_slip["avg_price"] * 100.0).round(1)
        disp_slip["exec_price"] = (disp_slip["exec_price"] * 100.0).round(1)

        edited_slip = st.data_editor(
            disp_slip,
            key=f"slip_editor_{st.session_state['editor_ver']}",
            use_container_width=True,
            hide_index=True,
            column_order=[
                "entered",
                "source_tag",
                "title",
                "outcome",
                "final_user_bet",
                "action_note",
                "avg_price",
                "exec_price",
                "slippage_cents",
                "max_limit_cents",
                "entry_cost_usdc",
                "last_buy_ago",
                "url",
            ],
            column_config={
                "entered": st.column_config.CheckboxColumn("✅ Đã kê Limit", default=False),
                "source_tag": st.column_config.TextColumn("Nguồn Ví", width="small"),
                "title": st.column_config.TextColumn("Trận đấu / Kèo", width="medium"),
                "outcome": st.column_config.TextColumn("Cửa chọn", width="small"),
                "final_user_bet": st.column_config.NumberColumn("Tiền bạn vào ($)", format="$%.2f"),
                "action_note": st.column_config.TextColumn("Ghi chú", width="medium"),
                "avg_price": st.column_config.NumberColumn("Giá ví gốc (¢)", format="%.1f¢"),
                "exec_price": st.column_config.NumberColumn("Best Ask (¢)", format="%.1f¢"),
                "slippage_cents": st.column_config.NumberColumn("Trượt giá (¢)", format="%+.1f¢"),
                "max_limit_cents": st.column_config.NumberColumn("🛑 Giá trần (¢)", format="%.1f¢"),
                "entry_cost_usdc": st.column_config.NumberColumn("Vốn ví gốc ($)", format="$%,.0f"),
                "last_buy_ago": st.column_config.TextColumn("Cập nhật", width="small"),
                "url": st.column_config.LinkColumn("Polymarket", display_text="Mở kèo ↗"),
            },
            disabled=[
                "pos_key",
                "source_tag",
                "title",
                "outcome",
                "final_user_bet",
                "action_note",
                "avg_price",
                "exec_price",
                "slippage_cents",
                "max_limit_cents",
                "entry_cost_usdc",
                "last_buy_ago",
                "url",
            ],
        )

        new_entered_slip = set(edited_slip.loc[edited_slip["entered"], "pos_key"].tolist())
        old_entered_slip = set(disp_slip.loc[disp_slip["entered"], "pos_key"].tolist())
        if new_entered_slip != old_entered_slip:
            updated_set = (st.session_state["entered_ids"] - old_entered_slip) | new_entered_slip
            st.session_state["entered_ids"] = updated_set
            persist_entered_ids(updated_set, active_keys_all)
            st.rerun()

    # --- MỤC 3: CẢNH BÁO VÍ GỐC ĐANG HEDGE 2 ĐẦU ---
    if not hedge_df.empty:
        st.subheader(f"🟠 3. Cảnh Báo Ví Gốc Đang Hedge 2 Đầu ({len(hedge_df)})")
        disp_hedge = hedge_df[
            ["source_tag", "title", "outcome", "entry_cost_usdc", "raw_user_bet", "avg_price", "cur_price", "hedge_note", "url"]
        ].copy()
        disp_hedge["avg_price"] = (disp_hedge["avg_price"] * 100.0).round(1)
        disp_hedge["cur_price"] = (disp_hedge["cur_price"] * 100.0).round(1)
        st.dataframe(
            disp_hedge,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_tag": "Nguồn Ví",
                "title": "Trận đấu",
                "outcome": "Cửa",
                "entry_cost_usdc": st.column_config.NumberColumn("Vốn ví gốc ($)", format="$%,.0f"),
                "raw_user_bet": st.column_config.NumberColumn("Quy đổi ($)", format="$%.2f"),
                "avg_price": st.column_config.NumberColumn("Giá mua (¢)", format="%.1f¢"),
                "cur_price": st.column_config.NumberColumn("Giá hiện tại (¢)", format="%.1f¢"),
                "hedge_note": "Lý do chặn",
                "url": st.column_config.LinkColumn("Polymarket", display_text="Mở kèo ↗"),
            },
        )

    # --- MỤC 4: BẢNG THEO DÕI RIÊNG GIẢI CHẶN (CCT EUROPE) ---
    if not blocked_df.empty:
        st.subheader(f"🚫 4. Bảng Theo Dõi Riêng: Giải Chặn — {cs2_blocked_kw} ({len(blocked_df)})")
        disp_blk = blocked_df[
            [
                "source_tag",
                "title",
                "outcome",
                "entry_cost_usdc",
                "raw_user_bet",
                "other_cond",
                "avg_price",
                "cur_price",
                "total_pnl",
                "url",
            ]
        ].copy()
        disp_blk["avg_price"] = (disp_blk["avg_price"] * 100.0).round(1)
        disp_blk["cur_price"] = (disp_blk["cur_price"] * 100.0).round(1)
        st.dataframe(
            disp_blk,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_tag": "Nguồn Ví",
                "title": "Trận đấu (Giải chặn)",
                "outcome": "Cửa",
                "entry_cost_usdc": st.column_config.NumberColumn("Vốn ví gốc ($)", format="$%,.0f"),
                "raw_user_bet": st.column_config.NumberColumn("Quy đổi ($)", format="$%.2f"),
                "other_cond": "Đánh giá điều kiện khác",
                "avg_price": st.column_config.NumberColumn("Giá ví gốc (¢)", format="%.1f¢"),
                "cur_price": st.column_config.NumberColumn("Giá hiện tại (¢)", format="%.1f¢"),
                "total_pnl": st.column_config.NumberColumn("PnL tạm tính ví gốc ($)", format="$%+,.2f"),
                "url": st.column_config.LinkColumn("Polymarket", display_text="Mở kèo ↗"),
            },
        )

    # --- MỤC 5: CÁC KÈO BỊ LOẠI BỞI BỘ LỌC ---
    if not rejected_df.empty:
        with st.expander(f"⚪ 5. Xem Danh Sách Kèo Đã Loại Bỏ ({len(rejected_df)})", expanded=False):
            disp_rej = rejected_df[
                ["source_tag", "title", "outcome", "entry_cost_usdc", "avg_price", "cur_price", "reject_reason"]
            ].copy()
            disp_rej["avg_price"] = (disp_rej["avg_price"] * 100.0).round(1)
            disp_rej["cur_price"] = (disp_rej["cur_price"] * 100.0).round(1)
            st.dataframe(
                disp_rej,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "source_tag": "Nguồn Ví",
                    "title": "Trận đấu",
                    "outcome": "Cửa",
                    "entry_cost_usdc": st.column_config.NumberColumn("Vốn ví gốc ($)", format="$%,.0f"),
                    "avg_price": st.column_config.NumberColumn("Giá mua (¢)", format="%.1f¢"),
                    "cur_price": st.column_config.NumberColumn("Giá hiện tại (¢)", format="%.1f¢"),
                    "reject_reason": "Lý do loại",
                },
            )
