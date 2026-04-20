"""NBA 业务工具函数。"""

import json

import streamlit as st
from langchain_openai import ChatOpenAI
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguestandingsv3,
    playercareerstats,
    teamdashboardbygeneralsplits,
    teaminfocommon,
)
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams


def get_json_llm():
    """创建一个被约束为 JSON 输出格式的聊天模型。"""
    return ChatOpenAI(
        temperature=0,
        base_url=st.session_state.openai_base_url,
        model=st.session_state.chat_model,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def parse_nba_query(question: str) -> dict:
    """把自然语言 NBA 问题解析成结构化字段。"""
    try:
        json_llm = get_json_llm()

        # Prompt: 从自然语言问题里提取球员、赛季和目标统计项。
        nba_parse_prompt = f"""
Extract structured NBA query info.

Return JSON only.

Fields:
- player_name
- season (format: 2024-25)
- stat_type (points, rebounds, assists, steals, blocks)

Question:
{question}
"""
        response = json_llm.invoke(nba_parse_prompt)
        return json.loads(response.content)
    except Exception as e:
        print("Parse error:", e)
        return {}


def nba_stats_tool(question: str) -> str:
    """根据问题内容，路由到对应的 NBA API 逻辑。"""
    question_lower = question.lower()

    # 如果问题明显是在问今天或当前的比赛结果，直接走 live scoreboard。
    if any(word in question_lower for word in ["live", "today", "scoreboard", "scores"]):
        try:
            games = scoreboard.ScoreBoard()
            data = games.get_dict()
            results = []
            for game in data["scoreboard"]["games"]:
                home = game["homeTeam"]["teamTricode"]
                away = game["awayTeam"]["teamTricode"]
                home_score = game["homeTeam"]["score"]
                away_score = game["awayTeam"]["score"]
                status = game["gameStatusText"]
                results.append(f"{away} {away_score} @ {home} {home_score} ({status})")
            return "\n".join(results) if results else "No live games found today."
        except Exception as e:
            return f"Error fetching live games: {e}"

    # 这一段先处理球队基础信息问题，例如城市、简称、建队时间。
    if any(word in question_lower for word in ["team", "franchise", "coach", "city", "state"]):
        try:
            all_teams = teams.get_teams()
            team_found = None

            for t in all_teams:
                full_name = t["full_name"].lower()
                city = t["city"].lower()
                nickname = t["nickname"].lower()
                abbr = t["abbreviation"].lower()

                if (
                    full_name in question_lower
                    or city in question_lower
                    or nickname in question_lower
                    or abbr in question_lower
                ):
                    team_found = t
                    break

            if team_found:
                return (
                    f"Team: {team_found['full_name']} ({team_found['abbreviation']})\n"
                    f"City: {team_found['city']}\n"
                    f"Founded: {team_found['year_founded']}"
                )

            return "No team found matching your query."
        except Exception as e:
            return f"Error fetching team info: {e}"

    # 更细的球队问题，例如战绩、排名、阵容、主场等，会调用更具体的 endpoint。
    if any(word in question_lower for word in ["team", "franchise", "coach", "city", "state", "record", "rank", "roster", "arena"]):
        try:
            all_teams = teams.get_teams()
            team_found = None

            for t in all_teams:
                if (
                    t["full_name"].lower() in question_lower
                    or t["city"].lower() in question_lower
                    or t["nickname"].lower() in question_lower
                    or t["abbreviation"].lower() in question_lower
                ):
                    team_found = t
                    break

            if not team_found:
                return "No team found matching your query."

            team_id = team_found["id"]
            team_name = team_found["full_name"]

            if "record" in question_lower or "wins" in question_lower:
                dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id=team_id)
                df = dashboard.get_data_frames()[0]
                wins = df.iloc[0]["W"]
                losses = df.iloc[0]["L"]
                return f"{team_name} current record: {wins}-{losses}"

            if "rank" in question_lower or "standing" in question_lower:
                standings = leaguestandingsv3.LeagueStandingsV3()
                df = standings.get_data_frames()[0]
                team_row = df[df["TeamID"] == team_id]
                if not team_row.empty:
                    rank = team_row.iloc[0]["PlayoffRank"]
                    conference = team_row.iloc[0]["Conference"]
                    return f"{team_name} is currently ranked #{rank} in the {conference} Conference."
                return "Ranking data not found."

            if "roster" in question_lower or "players" in question_lower:
                roster = commonteamroster.CommonTeamRoster(team_id=team_id)
                df = roster.get_data_frames()[0]
                player_list = df["PLAYER"].tolist()
                return f"{team_name} roster:\n" + "\n".join(player_list)

            if "coach" in question_lower or "arena" in question_lower:
                info = teaminfocommon.TeamInfoCommon(team_id=team_id)
                df = info.get_data_frames()[0]
                coach = df.iloc[0]["HEADCOACH"]
                arena = df.iloc[0]["ARENA"]
                return f"{team_name}\nHead Coach: {coach}\nHome Arena: {arena}"

            return (
                f"Team: {team_name}\n"
                f"City: {team_found['city']}\n"
                f"Founded: {team_found['year_founded']}"
            )
        except Exception as e:
            return f"Error fetching team info: {e}"

    try:
        # 如果前面都没命中，就按球员数据问题处理。
        all_players = nba_players.get_players()

        player_name = None
        for p in all_players:
            if p["full_name"].lower() in question.lower():
                player_name = p["full_name"]
                break

        if not player_name:
            return "No NBA player detected."

        parsed = parse_nba_query(question)
        season = parsed.get("season")
        stat_type = parsed.get("stat_type")
        if season and len(season) == 9:
            season = season[:4] + "-" + season[-2:]

        candidates = nba_players.find_players_by_full_name(player_name)
        if not candidates:
            return f"Player not found: {player_name}"

        player = candidates[0]
        endpoint = playercareerstats.PlayerCareerStats(player_id=player["id"])
        df = endpoint.get_data_frames()[0]

        if df.empty:
            return "No stats available."

        if season:
            df = df[df["SEASON_ID"].astype(str).str.contains(season)]

        if df.empty:
            return f"No data for season {season}"

        row = df.iloc[-1]
        gp = int(row["GP"])
        if gp == 0:
            return "No games played."

        stats_map = {
            "points": "PTS",
            "rebounds": "REB",
            "assists": "AST",
            "steals": "STL",
            "blocks": "BLK",
        }

        # 如果用户只问某一个统计项，就返回该项的场均值；
        # 否则给出一个简洁的 PTS / REB / AST 汇总。
        if stat_type and stat_type in stats_map:
            value = round(float(row[stats_map[stat_type]]) / gp, 2)
            return f"{player_name} averaged {value} {stat_type} per game in {row['SEASON_ID']}."

        pts = round(float(row["PTS"]) / gp, 2)
        reb = round(float(row["REB"]) / gp, 2)
        ast = round(float(row["AST"]) / gp, 2)
        return f"{player_name} averaged {pts} PPG, {reb} RPG, {ast} APG in {row['SEASON_ID']}."
    except Exception as e:
        return f"Error fetching stats: {e}"
