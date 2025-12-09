import streamlit as st
import pandas as pd
import numpy as np


# ページ設定（タイトル変更）
st.set_page_config(page_title="処理スコアダッシュボード📝", layout="wide")

# 左にロゴ、右にタイトル
logo_col, title_col = st.columns([1, 5])
with logo_col:
    # app.py と同じフォルダに「ロゴ_NEW_横.png」を置いてください
    st.image("ロゴ_NEW_横.png", use_column_width=True)
with title_col:
    st.title("処理スコアダッシュボード📝")

st.markdown(
    """
### 使い方
1. ①業務日報 ②業務割り振り ③案件管理 の3つのCSVをアップロード  
2. 集計対象の月を選択  
3. 各タブで「タスク別」「利用者別」「全体」を確認
"""
)


# ---------- 共通関数 ----------

def load_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile):
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
    st.error(f"ファイル {uploaded_file.name} を読み込めませんでした。エンコーディングをご確認ください。")
    return None


def assign_rank(row):
    """仕様に基づき 業務ランク(A〜E) を付与"""
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

    # ランクA（社外系を想定）
    if naigai:
        return "A"

    # どれにも該当しない場合は仮にE相当として扱う（必要に応じて変更）
    return "E"


RANK_VALUE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
RANK_ORDER_FOR_USER = {"E": 1, "D": 2, "C": 3, "B": 4, "A": 5}


def calc_deviation_by_task(df: pd.DataFrame, value_col: str, group_col: str = "task_id"):
    """同一タスクID内での偏差値（50±10）を計算"""
    group = df.groupby(group_col)[value_col]
    mean = group.transform("mean")
    std = group.transform("std").replace(0, np.nan)

    deviation = 50 + 10 * (df[value_col] - mean) / std
    deviation = deviation.fillna(50)  # 1人のみ / 全員同値 の場合は50点固定
    return deviation


def compute_all(report_df: pd.DataFrame, assign_df: pd.DataFrame,案件_df: pd.DataFrame, period_str: str):
    """
    3つのCSVから全ての集計を実施し、
    ・タスク別明細(base_df)
    ・利用者別集計(user_df)
    ・全体集計(summary_dict, org_summary_df)
    を返す
    """

    # ===== ① 業務日報：対象月の処理件数を集計 =====
    df_rep = report_df.copy()

    # 日付を変換
    df_rep["日付"] = pd.to_datetime(df_rep["日付"], errors="coerce")

    # 対象月のみ抽出（period_str は 'YYYY-MM'）
    target_period = pd.to_datetime(period_str + "-01")
    df_rep = df_rep[df_rep["日付"].dt.to_period("M") == target_period.to_period("M")]

    # タスクID/件数を正規化
    df_rep = df_rep[~df_rep["タスクID"].isna()]
    df_rep["タスクID"] = df_rep["タスクID"].astype(int)
    df_rep["件数"] = pd.to_numeric(df_rep["件数"], errors="coerce").fillna(0)

    # 利用者×タスクIDで月間件数
    monthly = (
        df_rep.groupby(["利用者コード", "タスクID"], as_index=False)["件数"]
        .sum()
        .rename(columns={"件数": "monthly_count"})
    )

    # ===== ② 業務割り振り × ③案件管理 を結合 =====
    df_assign = assign_df.copy()
    df案件 = 案件_df.copy()

    # 案件管理から必要な列のみ（タスクID/社内外/業務グループ）
    df案件 = df案件[["タスクID", "案件種別", "業務グループ"]].drop_duplicates()

    # 業務割り振りに案件情報を付与
    base = df_assign.merge(
        df案件,
        left_on="task_id",
        right_on="タスクID",
        how="left",
    )

    # 日報集計（月間件数）を付与
    base = base.merge(
        monthly,
        left_on=["employee_code", "task_id"],
        right_on=["利用者コード", "タスクID"],
        how="left",
    )

    base["monthly_count"] = base["monthly_count"].fillna(0)

    # ===== ③ ランク付与 =====
    base["rank"] = base.apply(assign_rank, axis=1)
    base["rank_value"] = base["rank"].map(RANK_VALUE).fillna(0)

    # ===== ④ 同一タスクID内で偏差値を計算 =====
    base["deviation"] = calc_deviation_by_task(base, "monthly_count", group_col="task_id")

    # ===== ⑤ 処理スコア = ランク値 × 処理偏差値 =====
    base["processing_score"] = base["rank_value"] * base["deviation"]

    # ----- 利用者別集計 -----
    user_df = (
        base.groupby(["employee_code", "user_name", "organization_name"], as_index=False)
        .agg(total_processing_score=("processing_score", "sum"))
    )

    # 各利用者が持っている業務ランク一覧（表示用）
    rank_list = (
        base.groupby(["employee_code", "user_name", "organization_name"])["rank"]
        .apply(lambda s: ", ".join(sorted(s.dropna().unique())))
        .reset_index(name="task_ranks")
    )
    user_df = user_df.merge(rank_list, on=["employee_code", "user_name", "organization_name"], how="left")

    # 自分自身の業務ランク（担当業務のうち最高位ランク）
    def best_rank(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return None
        order_values = s.map(RANK_ORDER_FOR_USER)
        return s.iloc[order_values.values.argmax()]

    user_rank = (
        base.groupby(["employee_code", "user_name", "organization_name"])["rank"]
        .apply(best_rank)
        .reset_index(name="user_rank")
    )
    user_df = user_df.merge(user_rank, on=["employee_code", "user_name", "organization_name"], how="left")

    # ----- 全体集計 -----
    overall_mean = user_df["total_processing_score"].mean()
    overall_median = user_df["total_processing_score"].median()

    summary_dict = {
        "overall_mean": overall_mean,
        "overall_median": overall_median,
    }

    # 拠点別：平均スコア & ランク構成（人数比）
    org_base = user_df.copy()
    org_summary = (
        org_base.groupby("organization_name", as_index=False)
        .agg(avg_score=("total_processing_score", "mean"), user_count=("employee_code", "nunique"))
    )

    rank_pivot = (
        org_base.pivot_table(
            index="organization_name",
            columns="user_rank",
            values="employee_code",
            aggfunc=pd.Series.nunique,
            fill_value=0,
        )
        .reset_index()
    )

    # 割合に変換
    org_summary = org_summary.merge(rank_pivot, on="organization_name", how="left")
    for r in ["A", "B", "C", "D", "E"]:
        if r in org_summary.columns:
            org_summary[f"ratio_{r}"] = org_summary[r] / org_summary["user_count"]
        else:
            org_summary[f"ratio_{r}"] = 0.0

    return base, user_df, summary_dict, org_summary


# ---------- ファイル入力 ----------

st.sidebar.header("1. CSVアップロード")

# ラベル文言をシンプルに修正
report_file = st.sidebar.file_uploader("① 業務日報CSV", type=["csv"])
assign_file = st.sidebar.file_uploader("② 業務割り振りCSV", type=["csv"])
案件_file = st.sidebar.file_uploader("③ 案件管理CSV", type=["csv"])

if not (report_file and assign_file and 案件_file):
    st.info("左のサイドバーから 3つのCSV をすべてアップロードしてください。")
    st.stop()

# 読み込み
df_report = load_csv(report_file)
df_assign = load_csv(assign_file)
df_案件 = load_csv(案件_file)

if df_report is None or df_assign is None or df_案件 is None:
    st.stop()

# ---------- 集計対象月の選択 ----------

# 日報データから年月一覧を作成
df_report["日付"] = pd.to_datetime(df_report["日付"], errors="coerce")
valid_dates = df_report["日付"].dropna()

if valid_dates.empty:
    st.error("日報の『日付』列が正しく読み込めていません。CSVのフォーマットをご確認ください。")
    st.stop()

periods = sorted(valid_dates.dt.to_period("M").astype(str).unique())
default_period = periods[-1]  # 最新月をデフォルト

selected_period = st.sidebar.selectbox("2. 集計対象の月", periods, index=periods.index(default_period))

st.sidebar.success(f"集計対象の月：{selected_period}")

# ---------- 集計実行 ----------

base_df, user_df, summary, org_summary_df = compute_all(
    df_report, df_assign, df_案件, selected_period
)

# ---------- 表示 ----------

tab1, tab2, tab3 = st.tabs(["タスク別処理状況", "利用者別集計", "全体集計"])

with tab1:
    st.subheader("タスク別処理状況（業務割り振りベース）")
    st.markdown(
        """
- `monthly_count`：その月の処理件数  
- `deviation`：同一タスクID内での処理件数の偏差値（平均50・標準偏差10）  
- `rank`：業務ランク（A=社外 / B=社内 / C=フォーム送信 / D=リスト作成・データ入力 / E=その他・練習 など）  
- `processing_score`：処理スコア = ランク値(A=5〜E=1) × 偏差値
"""
    )
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
    st.dataframe(base_df[show_cols].sort_values(["organization_name", "user_name", "task_id"]))

with tab2:
    st.subheader("利用者別 集計結果")
    st.markdown(
        """
- `task_ranks`：担当している業務ランクの一覧  
- `user_rank`：その利用者自身の業務ランク（担当業務の中で最も高いランク）  
- `total_processing_score`：担当業務の処理スコア合計
"""
    )
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
            ["organization_name", "total_processing_score"], ascending=[True, False]
        )
    )

with tab3:
    st.subheader(f"全体集計（{selected_period}）")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("処理スコア 平均値", f"{summary['overall_mean']:.2f}")
    with col2:
        st.metric("処理スコア 中央値", f"{summary['overall_median']:.2f}")

    st.markdown("### 拠点別 処理スコア平均 & ランク構成（人数比）")

    # 表示用に列を整理
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
    df_disp = org_summary_df[display_cols].copy()
    for c in ["ratio_A", "ratio_B", "ratio_C", "ratio_D", "ratio_E"]:
        df_disp[c] = (df_disp[c] * 100).round(1)  # %

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
