"""Web Mercantile – Warehouse Leasing Optimisation (Streamlit app)."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="Warehouse Leasing Optimiser", page_icon="🏬", layout="wide")
st.title("Web Mercantile – Warehouse Leasing Optimiser")
st.caption("Choose which leases to sign so every month's space requirement is met at minimum total cost.")

DEFAULT_REQ = [30000, 20000, 40000, 10000, 50000]
DEFAULT_COST = [65, 100, 135, 160, 190]

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Inputs")
    n = st.number_input("Planning horizon (months)", 1, 12, 5)
    integer = st.checkbox("Whole square feet only (integer)", value=False)
    st.markdown("Edit the tables to try other scenarios.")

req_default = pd.DataFrame({
    "Month": range(1, n + 1),
    "Required Space (sq ft)": [DEFAULT_REQ[i] if i < 5 else 0 for i in range(n)],
})
cost_default = pd.DataFrame({
    "Leasing Period (months)": range(1, n + 1),
    "Cost per Sq Ft ($)": [DEFAULT_COST[i] if i < 5 else DEFAULT_COST[-1] + 30 * (i - 4) for i in range(n)],
})

c1, c2 = st.columns(2)
with c1:
    st.subheader("Space requirements")
    req_df = st.data_editor(req_default, hide_index=True, disabled=["Month"], key=f"req{n}")
with c2:
    st.subheader("Leasing costs")
    cost_df = st.data_editor(cost_default, hide_index=True, disabled=["Leasing Period (months)"], key=f"cost{n}")

req = req_df["Required Space (sq ft)"].astype(float).to_numpy()
cost = cost_df["Cost per Sq Ft ($)"].astype(float).to_numpy()

# ---------------- Model ----------------
# Decision variable x[s,d] = sq ft leased starting in month s for d months (s + d - 1 <= n)
options = [(s, d) for s in range(1, n + 1) for d in range(1, n - s + 2)]
c = np.array([cost[d - 1] for s, d in options])
A = np.array([[1 if s <= t <= s + d - 1 else 0 for s, d in options] for t in range(1, n + 1)])

res = milp(
    c,
    constraints=LinearConstraint(A, lb=req),
    bounds=Bounds(0, np.inf),
    integrality=np.ones(len(c)) if integer else np.zeros(len(c)),
)

st.divider()
if not res.success:
    st.error(f"Solver could not find a solution: {res.message}")
    st.stop()

x = np.round(res.x, 4)
total = float(res.fun)

# ---------------- Results ----------------
m1, m2, m3 = st.columns(3)
m1.metric("Minimum total leasing cost", f"${total:,.0f}")
monthly_only = float((req * cost[0]).sum())
m2.metric("Cost if leasing month-by-month", f"${monthly_only:,.0f}", f"-${monthly_only - total:,.0f} saved", delta_color="inverse")
max_all = float(req.max() * cost[n - 1])
m3.metric(f"Cost if leasing the maximum for all {n} months", f"${max_all:,.0f}", f"-${max_all - total:,.0f} saved", delta_color="inverse")

st.subheader("Optimal leasing plan")
plan = pd.DataFrame(
    [
        {"Start Month": s, "Duration (months)": d, "End Month": s + d - 1,
         "Sq Ft Leased": x[i], "Cost per Sq Ft ($)": cost[d - 1], "Lease Cost ($)": x[i] * cost[d - 1]}
        for i, (s, d) in enumerate(options) if x[i] > 1e-6
    ]
)
st.dataframe(plan.style.format({"Sq Ft Leased": "{:,.0f}", "Cost per Sq Ft ($)": "${:,.0f}", "Lease Cost ($)": "${:,.0f}"}),
             hide_index=True, width="stretch")

st.subheader("Space leased vs required")
leased = A @ x
chart = pd.DataFrame({"Month": [f"Month {t}" for t in range(1, n + 1)], "Required": req, "Leased": leased}).set_index("Month")
st.bar_chart(chart, stack=False)
cov = chart.assign(Surplus=chart["Leased"] - chart["Required"])
st.dataframe(cov.style.format("{:,.0f}"), width="stretch")

with st.expander("Model formulation"):
    st.markdown(r"""
**Decision variables:** $x_{s,d}$ = square feet leased at the start of month $s$ for $d$ months ($s+d-1 \le n$).

**Objective:** minimise $\sum_{s,d} c_d\,x_{s,d}$, where $c_d$ is the cost per sq ft of a $d$-month lease.

**Constraints:** for every month $t$, $\sum_{s \le t \le s+d-1} x_{s,d} \ge R_t$ (space leased covers the requirement), and $x_{s,d} \ge 0$.

Solved as a linear programme with SciPy's HiGHS solver.
""")
