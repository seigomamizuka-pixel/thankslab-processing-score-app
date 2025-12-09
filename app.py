import streamlit as st
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# ページ設定（タイトル）
# ---------------------------------------------------------
st.set_page_config(page_title="処理スコアダッシュボード📝", layout="wide")

st.title("処理スコアダッシュボード📝")

st.markdown(
    """
### 使い方
1. ①業務日報（複数月分OK） ②業務割り振り ③案件管理 のCSVをアップロード  
2. 「詳細表示する月」で、タスク別 / 利用者別の表示対象の月を切り替え  
3. 「全体集計」タブで、月ごとの処理スコアを比較
"""
)


# ---------------------------------------------------------
# CSV 読み込み共通関数
# ---------------------------------------------------------
def load_csv(uploaded_file):
    """日本語CSVをいい感じのエンコーディングで読む"""
    if uploaded_file is None:
        return None

    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc, engine="python")
            return df
        except Exception:
            continue

    st.error(f"ファイル {uploaded_file.name} を読み込めませんでした。")
    return None


# ---------------------------------------------------------
# 業務ランク付与
# ---------------------------------------------------------
def assign_rank(row):
    status = str(row.get("task_status", ""))
    task_name = str(row.get("task_name", ""))
    genre = str(row.get("業務グループ", ""))
    naigai = str(row.get("案件種別", ""))

    # ランクE
    if (
        status == "練習"
        or task_name in [
            "【デモ】採用スカウト送信テスト",
            "【デモ】採用スカウトダミープロフィール作成業務",
            "その他",
        ]
        or genre in ["その他", "軽作業", "アンケート回答"]
    ):
        return "E"

    # ランクD
    if genre in ["リスト作成", "データ入力"]:
        return "D"

    # ランクC
    if genre == "フォーム送信":
        return "C"

    # ランクB
    if naigai == "社内BPO":
        return "B"

    # ランクA
    if naigai:
        return "A"

    return "E"


RANK_VALUE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
RANK_ORDER_FOR_USER = {"E": 1, "D": 2, "C": 3, "B": 4, "A": 5}


# ---------------------------------------------------------
# 偏差値計算
# ---------------------------------------------------------
def calc_deviation_by_task(df, value_col, group_col="task_id"):
    group = df.groupby(group_col)[value_col]
    mean = group.transform("mean")
    std = group.transform("std").replace(0, np.nan)

    deviation = 50 + 10 * (df[value_col] - mean) / std
    return deviation.fillna(50)


# ---------------------------------------------------------
# 1ヶ月分の全集計ロジック
# ---------------------------------------------------------
def compute_all(report_df, assign_df, 案件_df, period_str):
    """
    report_df：複数月分を含む日報全体
    period_str：'YYYY-MM' 形式の集計対象月
    """
    df_rep = report_df.copy()

    # 日付処理
    df_rep["日付"] = pd.to_datetime(df_rep["日付"], errors="coerce")
    target_period = pd.to_datetime(period_str + "-01")
    df_rep = df_rep[df_rep["日付"].dt.to_period("M") == target_period.to_period("M")]

    if df_rep.empty:
        # 対象月にデータがない場合は空の結果を返す
        empty_cols = ["employee_code", "user_name", "organization_name",
                      "task_id", "task_name", "task_status",
                      "案件種別", "業務グループ",
                      "rank", "rank_value", "monthly_count",
                      "deviation", "processing_score"]
        base_empty = pd.DataFrame(columns=empty_cols)

        user_empty = pd.DataFrame(
            columns=[
                "employee_code",
                "user_name",
                "organization_name",
                "total_processing_score",
                "task_ranks",
                "user_rank",
            ]
        )

        summary = {"overall_mean": np.nan, "overall_median": np.nan}
        org_summary = pd.DataFrame(
            columns=[
                "organization_name",
                "avg_score",
                "user_count",
                "ratio_A",
                "ratio_B",
                "ratio_C",
                "ratio_D",
                "ratio_E",
            ]
        )

        return base_empty, user_empty, summary, org_summary

    # 日報の集計
    df_rep = df_rep[~df_rep["タスクID"].isna()]
    df_rep["タスクID"] = df_rep["タスクID"].astype(int)
    df_rep["件数"] = pd.to_numeric(df_rep["件数"], errors="coerce").fillna(0)

    monthly = (
        df_rep.groupby(["利用者コード", "タスクID"], as_index=False)["件数"]
        .sum()
        .rename(columns={"件数": "monthly_count"})
    )

    # 業務割り振りと案件管理を結合
    df_assign2 = assign_df.copy()
    案件_df2 = 案件_df[["タスクID", "案件種別", "業務グループ"]].drop_duplicates()

    base = df_assign2.merge(
        案件_df2, left_on="task_id", right_on="タスクID", how="left"
    ).merge(
        monthly,
        left_on=["employee_code", "task_id"],
        right_on=["利用者コード", "タスクID"],
        how="left",
    )

    base["monthly_count"] = base["monthly_count"].fillna(0)

    # ランク付与
    base["rank"] = base.apply(assign_rank, axis=1)
    base["rank_value"] = base["rank"].map(RANK_VALUE)

    # 偏差値
    base["deviation"] = calc_deviation_by_task(base, "monthly_count")

    # 処理スコア
    base["processing_score"] = base["rank_value"] * base["deviation"]

    # -----------------------------------------------------
    # 利用者別集計
    # -----------------------------------------------------
    user_df = (
        base.groupby(["employee_code", "user_name", "organization_name"], as_index=False)
        .agg(total_processing_score=("processing_score", "sum"))
    )

    # 業務ランク一覧
    rank_list = (
        base.groupby(["employee_code", "user_name", "organization_name"])["rank"]
        .apply(lambda s: ", ".join(sorted(s.dropna().unique())))
        .reset_index(name="task_ranks")
    )

    user_df = user_df.merge(rank_list, on=["employee_code", "user_name", "organization_name"], how="left")

    # 個人最高ランク
    def best_rank(series):
        s = series.dropna()
        if s.empty:
            return None
        order_values = s.map(RANK_ORDER_FOR_USER)
        return s.iloc[order_values.values.argmax()]

    best_rank_df = (
        base.groupby(["employee_code", "user_name", "organization_name"])["rank"]
        .apply(best_rank)
        .reset_index(name="user_rank")
    )

    user_df = user_df.merge(best_rank_df, on=["employee_code", "user_name", "organization_name"], how="left")

    # -----------------------------------------------------
    # 全体集計
    # -----------------------------------------------------
    summary = {
        "overall_mean": user_df["total_processing_score"].mean(),
        "overall_median": user_df["total_processing_score"].median(),
    }

    org_summary = (
        user_df.groupby("organization_name", as_index=False)
        .agg(avg_score=("total_processing_score", "mean"), user_count=("employee_code", "nunique"))
    )

    rank_pivot = (
        user_df.pivot_table(
            index="organization_name",
            columns="user_rank",
            values="employee_code",
            aggfunc=pd.Series.nunique,
            fill_value=0,
        )
        .reset_index()
    )

    org_summary = org_summary.merge(rank_pivot, on="organization_name", how="left")

    for r in ["A", "B", "C", "D", "E"]:
        org_summary[f"ratio_{r}"] = org_summary.get(r, 0) / org_summary["user_count"]

    return base, user_df, summary, org_summary


# ---------------------------------------------------------
# CSV アップロード
# ---------------------------------------------------------
st.sidebar.header("1. CSVアップロード")

# ★日報だけ複数ファイルを許可★
report_files = st.sidebar.file_uploader(
    "① 業務日報CSV（1ヶ月1ファイル・複数月分アップロード可）",
    type=["csv"],
    accept_multiple_files=True,
)
assign_file = st.sidebar.file_uploader("② 業務割り振りCSV", type=["csv"])
案件_file = st.sidebar.file_uploader("③ 案件管理CSV", type=["csv"])

if not report_files or not assign_file or not 案件_file:
    st.info("左のサイドバーから ①〜③ すべてのCSVをアップロードしてください。")
    st.stop()

# 日報複数ファイルを結合
report_dfs = []
for f in report_files:
    df_tmp = load_csv(f)
    if df_tmp is not None:
        report_dfs.append(df_tmp)

if not report_dfs:
    st.error("業務日報CSVが読み込めませんでした。")
    st.stop()

df_report_all = pd.concat(report_dfs, ignore_index=True)

# その他2つ
df_assign = load_csv(assign_file)
df_案件 = load_csv(案件_file)

if df_assign is None or df_案件 is None:
    st.stop()

# ---------------------------------------------------------
# 利用可能な月一覧の取得
# ---------------------------------------------------------
df_report_all["日付"] = pd.to_datetime(df_report_all["日付"], errors="coerce")
valid_dates = df_report_all["日付"].dropna()

if valid_dates.empty:
    st.error("日報の『日付』列が読み込めていません。CSVをご確認ください。")
    st.stop()

periods = sorted(valid_dates.dt.to_period("M").astype(str).unique())

# 詳細表示する月（タスク別・利用者別用）
selected_period = st.sidebar.selectbox(
    "2. 詳細表示する月（タスク別・利用者別）", periods, index=len(periods) - 1
)

st.sidebar.write("※ 全体集計タブでは、アップロードされたすべての月を比較表示します。")

# ---------------------------------------------------------
# 各月ごとの集計をまとめて計算
# ---------------------------------------------------------
results_by_period = {}

for p in periods:
    base_df_p, user_df_p, summary_p, org_summary_p = compute_all(
        df_report_all, df_assign, df_案件, p
    )
    results_by_period[p] = {
        "base": base_df_p,
        "user": user_df_p,
        "summary": summary_p,
        "org_summary": org_summary_p,
    }

# 表示用に、選択された月のデータを取り出す
base_df = results_by_period[selected_period]["base"]
user_df = results_by_period[selected_period]["user"]
summary_selected = results_by_period[selected_period]["summary"]
org_summary_selected = results_by_period[selected_period]["org_summary"]


# ---------------------------------------------------------
# 表示タブ
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["タスク別処理状況", "利用者別集計", "全体集計"])


# ---------------------------------------------------------
# タスク別
# ---------------------------------------------------------
with tab1:
    st.subheader(f"タスク別処理状況（{selected_period}）")
    st.caption("※ サイドバーの月選択で切り替えられます。")

    show_cols = [
        "employee_code",
        "user_name",
        "organization_name",
        "task_id",
        "task_name",
        "task_status",
        "案件種別",
        "業務グループ",
        "rank",
        "rank_value",
        "monthly_count",
        "deviation",
        "processing_score",
    ]
    show_cols = [c for c in show_cols if c in base_df.columns]
    st.dataframe(
        base_df[show_cols].sort_values(
            ["organization_name", "user_name", "task_id"]
        )
    )


# ---------------------------------------------------------
# 利用者別
# ---------------------------------------------------------
with tab2:
    st.subheader(f"利用者別 集計結果（{selected_period}）")
    st.caption("※ サイドバーの月選択で切り替えられます。")

    show_cols = [
        "employee_code",
        "user_name",
        "organization_name",
        "user_rank",
        "task_ranks",
        "total_processing_score",
    ]
    st.dataframe(
        user_df[show_cols].sort_values(
            ["organization_name", "total_processing_score"],
            ascending=[True, False],
        )
    )


# ---------------------------------------------------------
# 全体集計（複数月比較）
# ---------------------------------------------------------
with tab3:
    st.subheader("全体集計（月比較）")

    # 月ごとの平均・中央値をまとめる
    summary_rows = []
    for p in periods:
        s = results_by_period[p]["summary"]
        summary_rows.append(
            {
                "month": p,
                "mean_score": s["overall_mean"],
                "median_score": s["overall_median"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("month")

    st.markdown("#### 月別 処理スコア（平均値・中央値）")
    st.dataframe(
        summary_df.assign(
            mean_score=lambda d: d["mean_score"].round(2),
            median_score=lambda d: d["median_score"].round(2),
        )
    )

    st.markdown("#### 拠点別 集計（選択月）")
    st.caption(f"※ 拠点別は現在選択中の月（{selected_period}）のみ表示")

    if not org_summary_selected.empty:
        display_cols = [
            "organization_name",
            "avg_score",
            "user_count",
            "ratio_A",
            "ratio_B",
            "ratio_C",
            "ratio_D",
            "ratio_E",
        ]

        df_disp = org_summary_selected[display_cols].copy()
        for c in ["ratio_A", "ratio_B", "ratio_C", "ratio_D", "ratio_E"]:
            df_disp[c] = (df_disp[c] * 100).round(1)

        df_disp = df_disp.rename(
            columns={
                "organization_name": "拠点",
                "avg_score": "平均スコア",
                "user_count": "利用者数",
                "ratio_A": "A比率(%)",
                "ratio_B": "B比率(%)",
                "ratio_C": "C比率(%)",
                "ratio_D": "D比率(%)",
                "ratio_E": "E比率(%)",
            }
        )

        st.dataframe(df_disp.sort_values("平均スコア", ascending=False))
    else:
        st.info(f"{selected_period} の拠点別集計データがありません。")
