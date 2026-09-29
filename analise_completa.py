# -*- coding: utf-8 -*-
"""
Reprocessamento completo do artigo "IA Generativa e Empreendedorismo Tecnológico" (v05).
Gera todas as tabelas, testes e a figura a partir dos CSV do GitHub Innovation Graph
e da planilha de abertura de empresas (Receita Federal via Base dos Dados).

Uso:  python analise_completa.py            -> versão corrigida (v05)
      python analise_completa.py --original -> reproduz os números do v04
Dependências: pandas, numpy, scipy, statsmodels, linearmodels, matplotlib, openpyxl
"""
import sys, json, os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

ORIGINAL = '--original' in sys.argv
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'resultados_v04_reproducao' if ORIGINAL else 'resultados_v05')
os.makedirs(OUT, exist_ok=True)
R = {}                                    # dicionário de resultados -> resultados.json

def rd(f):  # 'NA' (Namíbia) não pode ser lido como ausente
    return pd.read_csv(os.path.join(BASE, f), keep_default_na=False, na_values=[''])

VARS = ['organizations', 'git_pushes', 'developers', 'repositories']
LIM = 500                                 # corte de |crescimento| em %
ERA_CUT = (20223, 20251)                  # pré até 2022Q3; pós até 2025Q1; agentes a partir de 2025Q2

def growth(df, cols):
    """Crescimento ano sobre ano (%). v04: desloca 4 linhas; v05: desloca 4 trimestres de calendário."""
    df = df.sort_values(['iso2_code', 'year', 'quarter']).copy()
    if ORIGINAL:
        for v in cols:
            df[v + '_growth'] = df.groupby('iso2_code')[v].pct_change(4, fill_method=None) * 100
        return df
    df['t'] = df.year * 4 + df.quarter
    for v in cols:
        l = df[['iso2_code', 't', v]].copy(); l['t'] += 4
        df = df.merge(l.rename(columns={v: v + '_l4'}), on=['iso2_code', 't'], how='left')
        df[v + '_growth'] = (df[v] / df[v + '_l4'] - 1) * 100
    return df.drop(columns='t')

def add_lag(df, v):
    df = df.copy(); df['t'] = df.year * 4 + df.quarter
    l = df[['iso2_code', 't', v]].copy(); l['t'] += 1
    df = df.merge(l.rename(columns={v: v + '_lag1'}), on=['iso2_code', 't'], how='left')
    return df.drop(columns='t')

def era(df):
    q = df.year * 10 + df.quarter
    return np.where(q <= ERA_CUT[0], 'pre', np.where(q <= ERA_CUT[1], 'pos', 'agentes'))

def idx(df):
    df = df.copy()
    df['per'] = pd.PeriodIndex(df.year.astype(str) + 'Q' + df.quarter.astype(str), freq='Q').to_timestamp()
    return df.set_index(['iso2_code', 'per'])

def clean(df, y, x):
    d = df.dropna(subset=[y, x])
    return d[(d[y].abs() <= LIM) & (d[x].abs() <= LIM)].copy()

def twfe(d, y, xs):
    return PanelOLS.from_formula(f'{y} ~ {"+".join(xs)} + EntityEffects + TimeEffects', idx(d)).fit(
        cov_type='clustered', cluster_entity=True)

def res(m, x):
    return dict(beta=round(float(m.params[x]), 4), se=round(float(m.std_errors[x]), 4),
                p=round(float(m.pvalues[x]), 4), n=int(m.nobs), economias=int(m.entity_info['total']),
                r2_within=round(float(m.rsquared_within), 4), f_pool=round(float(m.f_pooled.stat), 2), p_pool=float(m.f_pooled.pval))

# ------------------------------------------------------------------ 1. Série agregada global (níveis)
fr = [rd(f + '.csv').set_index(['iso2_code', 'year', 'quarter'])[f] for f in VARS]
lev = pd.concat(fr, axis=1).reset_index()
agg = lev[lev.iso2_code != 'EU'].groupby(['year', 'quarter'])[VARS].sum()
R['agregado'] = {v: dict(media=round(agg[v].mean()), cv=round(agg[v].std() / agg[v].mean() * 100, 1)) for v in VARS}
R['agregado']['n_trimestres'] = len(agg)

# ------------------------------------------------------------------ 2. Painel principal
p = rd('panel_by_country.csv')
cov = p.groupby('iso2_code').size()
p = p[p.iso2_code.isin(cov[cov >= 20].index)]
R['economias_20trim'] = int(p.iso2_code.nunique())
g = growth(p, VARS)
g = add_lag(g, 'git_pushes_growth')
d = clean(g, 'organizations_growth', 'git_pushes_growth')
d['era'] = era(d)
R['painel_principal'] = dict(n=len(d), economias=int(d.iso2_code.nunique()))
R['descritivas'] = {v: dict(media=round(d[v].mean(), 2), dp=round(d[v].std(), 2), assim=round(stats.skew(d[v]), 2),
                            curtose=round(stats.kurtosis(d[v]), 2))
                    for v in ['organizations_growth', 'git_pushes_growth']}

# correlações (Tabela 1): mesma amostra, pares completos das quatro taxas
c4 = d.dropna(subset=[v + '_growth' for v in VARS])
c4 = c4[(c4[[v + '_growth' for v in VARS]].abs() <= LIM).all(axis=1)]
R['tabela1'] = dict(n=len(c4),
                    pearson=c4[[v + '_growth' for v in VARS]].corr().round(3).to_dict(),
                    spearman=c4[[v + '_growth' for v in VARS]].corr('spearman').round(3).to_dict())
R['corr_principal'] = dict(pearson=[round(x, 4) for x in stats.pearsonr(d.organizations_growth, d.git_pushes_growth)],
                           spearman=[round(x, 4) for x in stats.spearmanr(d.organizations_growth, d.git_pushes_growth)])

m0 = twfe(d, 'organizations_growth', ['git_pushes_growth'])
R['fe_contemporaneo'] = res(m0, 'git_pushes_growth')
R['poolability'] = dict(stat=round(float(m0.f_pooled.stat), 3), p=float(m0.f_pooled.pval))
dl = clean(d, 'organizations_growth', 'git_pushes_growth_lag1')
m1 = twfe(dl, 'organizations_growth', ['git_pushes_growth_lag1'])
R['fe_defasado'] = res(m1, 'git_pushes_growth_lag1')

# ------------------------------------------------------------------ 3. Interação por era + Wald
for e in ['pre', 'pos', 'agentes']:
    d['x_' + e] = d.git_pushes_growth * (d.era == e)
me = twfe(d, 'organizations_growth', ['x_pre', 'x_pos', 'x_agentes'])
R['eras'] = {e: res(me, 'x_' + e) for e in ['pre', 'pos', 'agentes']}
b, V = me.params, me.cov
def wald(r):  # r: vetor de restrição R b = 0
    r = np.asarray(r, float); dif = r @ b.values; var = r @ V.values @ r
    w = dif ** 2 / var; return dict(chi2=round(float(w), 3), p=round(float(stats.chi2.sf(w, 1)), 4))
R['wald'] = {'agentes_vs_pre': wald([-1, 0, 1]), 'pos_vs_pre': wald([-1, 1, 0]), 'agentes_vs_pos': wald([0, -1, 1])}
Rm = np.array([[-1, 1, 0], [-1, 0, 1]], float); dv = Rm @ b.values
wj = float(dv @ np.linalg.inv(Rm @ V.values @ Rm.T) @ dv)
R['wald']['conjunto_2gl'] = dict(chi2=round(wj, 3), p=round(float(stats.chi2.sf(wj, 2)), 4))

# sensibilidade do corte da era dos agentes
sens = {}
for cut in [20244, 20251, 20252, 20253, 20254]:
    q = d.year * 10 + d.quarter
    ee = np.where(q <= ERA_CUT[0], 'pre', np.where(q < cut, 'pos', 'agentes'))
    dd = d.copy()
    for e in ['pre', 'pos', 'agentes']: dd['x_' + e] = dd.git_pushes_growth * (ee == e)
    mm = twfe(dd, 'organizations_growth', ['x_pre', 'x_pos', 'x_agentes'])
    sens[str(cut)[:4] + 'Q' + str(cut)[-1]] = dict(loglik=round(float(mm.loglik), 1), beta_agentes=round(float(mm.params['x_agentes']), 4))
R['sensibilidade_corte'] = sens

# médias por era
mer = d.groupby('era')[['organizations_growth', 'git_pushes_growth']].mean().round(2)
R['medias_era'] = mer.to_dict()
t_ = stats.ttest_ind(d[d.era == 'pre'].organizations_growth, d[d.era == 'agentes'].organizations_growth, equal_var=False)
R['ttest_org_pre_vs_agentes'] = dict(t=round(float(t_.statistic), 3), p=round(float(t_.pvalue), 4))
t_ = stats.ttest_ind(d[d.era == 'pre'].git_pushes_growth, d[d.era == 'agentes'].git_pushes_growth, equal_var=False)
R['ttest_push_pre_vs_agentes'] = dict(t=round(float(t_.statistic), 3), p=round(float(t_.pvalue), 4))

# ------------------------------------------------------------------ 4. Hausman (FE x RE)
def hausman(x):
    dd = clean(d if x == 'git_pushes_growth' else g, 'organizations_growth', x)
    if ORIGINAL:
        return None
    di = idx(dd)
    td = pd.get_dummies(di.index.get_level_values(1), prefix='q', drop_first=True, dtype=float)
    td.index = di.index
    X = pd.concat([di[[x]], td], axis=1)
    fe_ = PanelOLS(di.organizations_growth, X, entity_effects=True).fit(cov_type='unadjusted')
    re_ = RandomEffects(di.organizations_growth, sm.add_constant(X)).fit(cov_type='unadjusted')
    fe_c = PanelOLS(di.organizations_growth, X, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    re_c = RandomEffects(di.organizations_growth, sm.add_constant(X)).fit(cov_type='clustered', cluster_entity=True)
    cols = [x] + list(td.columns)
    # Hausman clássico sobre o coeficiente de interesse, com covariâncias convencionais
    # (sob H0 o estimador RE precisa ser eficiente, o que não vale com erro clusterizado)
    H1 = float((fe_.params[x] - re_.params[x]) ** 2 / (fe_.cov.loc[x, x] - re_.cov.loc[x, x]))
    return dict(n=int(fe_.nobs), economias=int(fe_.entity_info['total']),
                beta_fe=round(float(fe_.params[x]), 4), se_fe_cluster=round(float(fe_c.std_errors[x]), 4), p_fe_cluster=round(float(fe_c.pvalues[x]), 4),
                beta_re=round(float(re_.params[x]), 4), se_re_cluster=round(float(re_c.std_errors[x]), 4), p_re_cluster=round(float(re_c.pvalues[x]), 4),
                theta_media=round(float(np.mean(re_.theta)), 3),
                hausman=round(H1, 3), gl=1, p_hausman=round(float(stats.chi2.sf(H1, 1)), 4))
if not ORIGINAL:
    R['hausman'] = {x: hausman(x) for x in ['git_pushes_growth', 'developers_growth', 'repositories_growth']}

# ------------------------------------------------------------------ 5. Robustez: tópicos de IA generativa e linguagens
GENAI = sorted("""llm llms large-language-models generative-ai ai artificial-intelligence chatgpt openai claude claude-code
codex gemini gemini-api ollama langchain rag mcp mcp-server model-context-protocol ai-agents ai-agent agentic-ai agents
agent multi-agent agent-skills openclaw ai-tools chatbot vibe-coding github-copilot""".split())
R['topicos_lista'] = GENAI
o = rd('organizations.csv')
tp = rd('topics.csv')
tp = tp[tp.topic.isin(GENAI) & (tp.iso2_code != 'EU')]
ag = tp.groupby(['iso2_code', 'year', 'quarter']).num_pushers.agg(ai_topic_engagement='sum', ai_topic_max='max').reset_index()
cv = ag.groupby('iso2_code').size(); ag = ag[ag.iso2_code.isin(cv[cv >= 12].index)]
ag = add_lag(add_lag(growth(ag.merge(o, on=['iso2_code', 'year', 'quarter']),
                            ['ai_topic_engagement', 'ai_topic_max', 'organizations']),
                     'ai_topic_engagement_growth'), 'ai_topic_max_growth')
R['topicos_economias'] = sorted(ag.iso2_code.unique())
rob = {}
for v in ['ai_topic_engagement_growth', 'ai_topic_max_growth']:
    for suf, lab in [('', 'contemporanea'), ('_lag1', 'defasada')]:
        dd = clean(ag, 'organizations_growth', v + suf)
        rob[f'{v}|{lab}'] = res(twfe(dd, 'organizations_growth', [v + suf]), v + suf)
la = rd('languages.csv')
la = la[la.language.isin(['Python', 'TypeScript', 'JavaScript']) & (la.iso2_code != 'EU')]
la = la.groupby(['iso2_code', 'year', 'quarter']).num_pushers.sum().rename('languages').reset_index()
cv = la.groupby('iso2_code').size(); la = la[la.iso2_code.isin(cv[cv >= 20].index)]
la = add_lag(growth(la.merge(o, on=['iso2_code', 'year', 'quarter']), ['languages', 'organizations']), 'languages_growth')
for suf, lab in [('', 'contemporanea'), ('_lag1', 'defasada')]:
    dd = clean(la, 'organizations_growth', 'languages_growth' + suf)
    rob[f'languages_growth|{lab}'] = res(twfe(dd, 'organizations_growth', ['languages_growth' + suf]), 'languages_growth' + suf)
R['robustez'] = rob

# ------------------------------------------------------------------ 6. Estudo de caso Brasil
bz = pd.read_excel(os.path.join(BASE, 'Abertura de Empresas Brasil.xlsx'), 'dados_cnae')
bz = bz[bz.mes < '2026-07-01']; bz['year'] = bz.mes.dt.year; bz['quarter'] = bz.mes.dt.quarter
CL = {'Tecnologia da informação': [62, 63], 'Marketing e conteúdo': [58, 59, 60, 73],
      'Serviços profissionais': [69, 70, 71, 72, 74] if not ORIGINAL else [62, 63, 69, 70, 71, 72, 73, 74],
      'Educação': [85], 'Comércio (controle)': [45, 46, 47], 'Construção civil (controle)': [41, 42, 43]}
def serie(divs, estr=None):
    x = bz[bz.cnae_divisao.isin(divs)]
    if estr: x = x[x.estrutura == estr]
    s = x.groupby(['year', 'quarter']).empresas_abertas.sum().reset_index(); s['iso2_code'] = 'BR'
    return s
tech = growth(serie([62, 63]), ['empresas_abertas']).rename(columns={'empresas_abertas_growth': 'tech_growth'})
solo = serie([62, 63], 'solo').rename(columns={'empresas_abertas': 'solo'})
tech = tech.merge(solo[['year', 'quarter', 'solo']], on=['year', 'quarter'])
tech['share_solo'] = tech.solo / tech.empresas_abertas * 100
brg = g[g.iso2_code == 'BR'][['year', 'quarter', 'organizations_growth', 'git_pushes_growth']]
mb = tech.merge(brg, on=['year', 'quarter']).dropna(subset=['tech_growth', 'organizations_growth'])
def hac(y, x, lags=4):
    z = (mb[x] - mb[x].mean()) / mb[x].std(); w = (mb[y] - mb[y].mean()) / mb[y].std()
    f = sm.OLS(w, sm.add_constant(z)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return round(float(f.pvalues.iloc[1]), 4)
R['brasil'] = dict(
    n=len(mb),
    org_vs_tech=dict(r=round(stats.pearsonr(mb.tech_growth, mb.organizations_growth)[0], 3), p=round(stats.pearsonr(mb.tech_growth, mb.organizations_growth)[1], 4), p_hac=hac('tech_growth', 'organizations_growth')),
    push_vs_tech=dict(r=round(stats.pearsonr(mb.tech_growth, mb.git_pushes_growth)[0], 3), p=round(stats.pearsonr(mb.tech_growth, mb.git_pushes_growth)[1], 4),
                      rho=round(stats.spearmanr(mb.tech_growth, mb.git_pushes_growth)[0], 3), p_rho=round(stats.spearmanr(mb.tech_growth, mb.git_pushes_growth)[1], 4), p_hac=hac('tech_growth', 'git_pushes_growth')),
    push_vs_solo=dict(r=round(stats.pearsonr(mb.share_solo, mb.git_pushes_growth)[0], 3), p=round(stats.pearsonr(mb.share_solo, mb.git_pushes_growth)[1], 4)))
mb['era'] = era(mb)
R['brasil']['era_amostra21'] = mb.groupby('era')[['tech_growth', 'git_pushes_growth', 'share_solo']].mean().round(2).to_dict()
allb = serie(list(bz.cnae_divisao.unique()));  sol = serie(list(bz.cnae_divisao.unique()), 'solo')
allb['share'] = sol.empresas_abertas / allb.empresas_abertas * 100; allb['era'] = era(allb)
R['brasil']['share_solo_geral_era'] = allb.groupby('era').share.mean().round(1).to_dict()
G = pd.DataFrame({k: growth(serie(v), ['empresas_abertas']).set_index(['year', 'quarter']).empresas_abertas_growth for k, v in CL.items()}).dropna()
G['era'] = era(G.reset_index())
R['tabela5'] = dict(n=len(G), medias=G.groupby('era').mean().round(2).to_dict())
Gc = G.drop(columns='era')
R['clusters_corr_niveis'] = Gc.corr().round(2).to_dict()
R['clusters_corr_diff'] = Gc.diff().dropna().corr().round(2).to_dict()   # primeiras diferenças: remove tendência comum

# ------------------------------------------------------------------ 7. Figura 1
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
lab = {'pre': 'Pré-ChatGPT\n(até 2022Q3)', 'pos': 'Pós-ChatGPT\n(2022Q4 a 2025Q1)', 'agentes': 'Era dos agentes\n(desde 2025Q2)'}
fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
xs = np.arange(3); w = 0.36
for i, (v, nm, col) in enumerate([('organizations_growth', 'organizations_growth', '#4a4a4a'), ('git_pushes_growth', 'git_pushes_growth', '#a8a8a8')]):
    vals = [mer.loc[e, v] for e in ['pre', 'pos', 'agentes']]
    bars = ax.bar(xs + (i - .5) * w, vals, w, label=nm, color=col, edgecolor='black', linewidth=.5)
    for bb, val in zip(bars, vals): ax.text(bb.get_x() + bb.get_width() / 2, val + 1, f'{val:.1f}'.replace('.', ','), ha='center', fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([lab[e] for e in ['pre', 'pos', 'agentes']], fontsize=9)
ax.set_ylabel('Crescimento médio ano sobre ano (%)', fontsize=9)
leg = ax.legend(fontsize=9, frameon=False)
for tx in leg.get_texts(): tx.set_fontstyle('italic')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figura1_crescimento_por_era.png'), dpi=300)

json.dump(R, open(os.path.join(OUT, 'resultados.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
