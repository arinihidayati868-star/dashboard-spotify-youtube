import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

HIJAU, MERAH, ABU = "#1DB954", "#E5383B", "#8D99AE"

st.set_page_config(page_title="Apa yang Membuat Lagu Populer?",
                   page_icon="🎵", layout="wide")

@st.cache_data
def muat_data():
    df = pd.read_csv("spotify_youtube_clean.csv")
    if "Lagu_Unik" not in df.columns:
        df["Lagu_Unik"] = True
    df["Lagu_Unik"] = df["Lagu_Unik"].astype(bool)
    # Hit = 25% stream teratas, dihitung dari lagu unik
    batas = df.loc[df["Lagu_Unik"], "Stream"].quantile(0.75)
    df["Kategori"] = np.where(df["Stream"] >= batas, "Hit (top 25%)", "Non-hit")
    df.loc[df["Stream"].isna(), "Kategori"] = np.nan
    for k in ["Video_Resmi", "Berlisensi"]:
        df[k] = df[k].fillna("Tidak ada data YouTube")
    return df


def juta(x):
    return "-" if pd.isna(x) else f"{x / 1e6:,.1f} jt"


df = muat_data()

# ---------- Sidebar: filter ----------
st.sidebar.header("Filter data")
f_artis = st.sidebar.multiselect("Artis", sorted(df["Artist"].unique()))
tipe_semua = sorted(df["Tipe_Album"].dropna().unique())
f_album = st.sidebar.multiselect("Tipe album", tipe_semua, default=tipe_semua)
video_semua = sorted(df["Video_Resmi"].unique())
f_video = st.sidebar.multiselect("Video resmi", video_semua, default=video_semua)
maks = float(df["Stream"].max() / 1e6)
f_stream = st.sidebar.slider("Rentang stream (juta)", 0.0, maks, (0.0, maks))

mask = (df["Tipe_Album"].isin(f_album) & df["Video_Resmi"].isin(f_video)
        & (df["Stream"].isna() | df["Stream"].between(f_stream[0] * 1e6, f_stream[1] * 1e6)))
if f_artis:
    mask &= df["Artist"].isin(f_artis)

d_all = df[mask]                       # level artis (semua baris)
d_lagu = d_all[d_all["Lagu_Unik"]]     # level lagu (tanpa duplikat kolaborasi)

if d_lagu.empty:
    st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

# ---------- Judul ----------
st.title("Apa yang Membuat Lagu Populer?")
st.caption("Analisis karakter audio dan performa lagu di Spotify dan YouTube")

# ---------- 3 KPI ----------
sv = d_lagu[["Stream", "Views"]].dropna()
rho = sv.corr(method="spearman").iloc[0, 1] if len(sv) > 2 else np.nan
k1, k2, k3 = st.columns(3)
k1.metric("Lagu unik", f"{len(d_lagu):,}")
k2.metric("Median stream Spotify", juta(d_lagu["Stream"].median()))
k3.metric("Korelasi Stream-Views", "-" if pd.isna(rho) else f"{rho:.2f}")

audio = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
         "Instrumentalness", "Liveness", "Valence", "Tempo", "Durasi_menit"]
skala01 = ["Danceability", "Energy", "Speechiness", "Acousticness",
           "Instrumentalness", "Liveness", "Valence"]

# ---------- Baris 1: Q1 (visualisasi 1 dan 2) ----------
st.subheader("Karakter audio apa yang dimiliki lagu populer?")
c1, c2 = st.columns(2)

kor = (d_lagu[audio + ["Stream"]].corr(method="spearman")["Stream"]
       .drop("Stream").sort_values().reset_index())
kor.columns = ["Fitur", "Korelasi"]
kor["Arah"] = np.where(kor["Korelasi"] >= 0, "Positif", "Negatif")
fig = px.bar(kor, x="Korelasi", y="Fitur", orientation="h", color="Arah",
             color_discrete_map={"Positif": HIJAU, "Negatif": MERAH},
             title="Korelasi fitur audio dengan Stream (Spearman)")
fig.update_layout(showlegend=False)
c1.plotly_chart(fig, use_container_width=True)

hn = (d_lagu.dropna(subset=["Kategori"]).groupby("Kategori")[skala01]
      .median().T.reset_index().melt(id_vars="index", var_name="Kategori", value_name="Median"))
fig = px.bar(hn, x="index", y="Median", color="Kategori", barmode="group",
             color_discrete_map={"Hit (top 25%)": HIJAU, "Non-hit": ABU},
             labels={"index": "Fitur"}, title="Median fitur audio: hit vs non-hit")
c2.plotly_chart(fig, use_container_width=True)

# ---------- Baris 2: Q2 dan Q3 (visualisasi 3 dan 4) ----------
st.subheader("Spotify vs YouTube: populer di mana?")
c1, c2 = st.columns(2)

sc = d_lagu.dropna(subset=["Log_Stream", "Log_Views"])
fig = px.scatter(sc, x="Log_Stream", y="Log_Views", color="Tipe_Album",
                 hover_data=["Artist", "Track"], opacity=0.5, render_mode="webgl",
                 labels={"Log_Stream": "Stream (log10)", "Log_Views": "Views YouTube (log10)",
                         "Tipe_Album": "Tipe album"},
                 title="Stream vs Views (skala log)")
c1.plotly_chart(fig, use_container_width=True)

art = (d_all.dropna(subset=["Stream", "Views"]).groupby("Artist")
       .agg(Total_Stream=("Stream", "sum"), Total_Views=("Views", "sum")).reset_index())
art["Views_per_Stream"] = (art["Total_Views"] / art["Total_Stream"]).round(2)
top = art.sort_values("Total_Stream", ascending=False).head(10).sort_values("Views_per_Stream")
fig = px.bar(top, x="Views_per_Stream", y="Artist", orientation="h",
             hover_data=["Total_Stream", "Total_Views"], color_discrete_sequence=[MERAH],
             labels={"Views_per_Stream": "Views per stream"},
             title="Views per stream, 10 artis stream terbesar")
fig.add_vline(x=1, line_dash="dash", line_color=ABU)
c2.plotly_chart(fig, use_container_width=True)
c2.caption("Di kanan garis putus-putus: condong ke YouTube. Di kirinya: condong ke Spotify. "
           "Total stream artis dihitung dari lagu teratasnya di dataset (maksimal 10 lagu).")

# ---------- Baris 3: Q4 (visualisasi 5) + insight ----------
st.subheader("Engagement dan rekomendasi")
c1, c2 = st.columns(2)

eng = d_lagu.dropna(subset=["Like_rate"]).copy()
eng["Like_rate_pct"] = eng["Like_rate"] * 100
a = eng.groupby("Tipe_Album")["Like_rate_pct"].median().round(2).reset_index()
fig = px.bar(a, x="Tipe_Album", y="Like_rate_pct", text="Like_rate_pct",
             color_discrete_sequence=[HIJAU],
             labels={"Like_rate_pct": "Median like rate (%)", "Tipe_Album": "Tipe album"},
             title="Like rate per tipe album")
c1.plotly_chart(fig, use_container_width=True)

# CATATAN: teks insight ditulis dari hasil EDA. Sesuaikan angkanya dengan hasil analismu.
c2.info(
    "**Key insight**\n\n"
    "- Karakter audio hanya sedikit membedakan lagu hit dari lainnya. Lagu hit cenderung "
    "lebih keras dan kurang akustik.\n"
    "- Popularitas di Spotify dan YouTube berkaitan, tetapi kecenderungan platform berbeda per artis.\n"
    "- Single punya like rate lebih tinggi (sekitar 1,3% vs 0,8% pada album), tetapi median stream "
"jauh lebih rendah (sekitar 30 juta vs 57 juta)."
)
c2.success(
    "**Recommendation**\n\n"
    "- Jangan mengandalkan formula audio semata. Artis dan promosi kemungkinan lebih berpengaruh.\n"
    "- Artis dengan views per stream rendah dapat memperkuat kehadiran di YouTube.\n"
    "- Gunakan single untuk membangun engagement, dan album untuk menjangkau lebih banyak pendengar."
)

st.caption("Sumber data: Spotify and YouTube (Kaggle, salvatorerastelli). Data hanya memuat lagu populer, "
           "tanpa informasi waktu rilis. Korelasi tidak menunjukkan sebab-akibat.")
