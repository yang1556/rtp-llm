# Gated Delta Rule: Context Parallelism 推导

## 1. 原始 Chunk-wise 递推

Gated Delta Rule 的 chunk-wise 前向计算分为 6 步：

```
Step 1: g = chunk_local_cumsum(g)
Step 2: A = chunk_scaled_dot_kkt(k, beta, g)
Step 3: A = solve_tril(A)
Step 4: w, u = recompute_w_u(k, v, beta, A, g)
Step 5: h, v_new = chunk_fwd_h(k, w, u, g, h0)       ← 串行瓶颈
Step 6: o = chunk_fwd_o(q, k, v_new, h, g, scale)
```

其中 Step 1-4 和 Step 6 的每个 chunk 独立可并行，Step 5 是跨 chunk 的串行递推。

### Step 5 递推公式

设序列被切分为 $NT$ 个 chunk，每个 chunk 大小为 $BT$。对第 $t$ 个 chunk：

$$
v_{new}[t] = v[t] - w[t] \cdot h[t] \quad \in \mathbb{R}^{BT \times V}
$$

$$
h[t+1] = e^{g^{last}_t} \cdot h[t] + k[t]^\top \cdot G_t \cdot v_{new}[t] \quad \in \mathbb{R}^{K \times V}
$$

其中：
- $h[t] \in \mathbb{R}^{K \times V}$：第 $t$ 个 chunk 的隐藏状态
- $w[t] \in \mathbb{R}^{BT \times K}$：WY 表示中的 $w$ 矩阵
- $k[t] \in \mathbb{R}^{BT \times K}$：key 矩阵（转置后为 $K \times BT$）
- $G_t = \text{diag}(e^{g^{last}_t - g[0..BT-1]}) \in \mathbb{R}^{BT \times BT}$：chunk 内 gating 对角矩阵
- $g^{last}_t$：第 $t$ 个 chunk 最后一个位置的 cumsum gate 值

### Step 6 输出公式

$$
o[t] = \underbrace{q[t] \cdot h[t] \cdot e^{g[t]} \cdot s}_{\text{inter-chunk}} + \underbrace{A^{causal}_t \cdot v_{new}[t] \cdot s}_{\text{intra-chunk}}
$$

其中：
- $s = 1/\sqrt{K}$：scale factor
- $A^{causal}_t = \text{tril}(q[t] \cdot k[t]^\top \cdot \text{gate})$：下三角 causal attention 矩阵

---

## 2. 仿射递推形式

将 $v_{new}[t]$ 代入 $h[t+1]$ 的更新公式：

$$
h[t+1] = e^{g^{last}_t} \cdot h[t] + k[t]^\top \cdot G_t \cdot (v[t] - w[t] \cdot h[t])
$$

$$
= \underbrace{\left(e^{g^{last}_t} \cdot I_K - k[t]^\top \cdot G_t \cdot w[t]\right)}_{M[t] \in \mathbb{R}^{K \times K}} \cdot h[t] + \underbrace{k[t]^\top \cdot G_t \cdot v[t]}_{b[t] \in \mathbb{R}^{K \times V}}
$$

即：

$$
\boxed{h[t+1] = M[t] \cdot h[t] + b[t]}
$$

这是一个仿射递推（affine recurrence），转移矩阵 $M[t]$ 是 $K \times K$ 矩阵，不是标量。

---

## 3. CP 并行策略：投机执行 + 修正

### 3.1 序列切分

将长度为 $T$ 的序列均分到 $P$ 个 rank，每个 rank 持有 $T_{local} = T / P$ 的连续片段。

### 3.2 投机执行

所有 rank 同时执行 Step 1-6：
- Rank 0 使用真实初始状态 $h_0$
- Rank $r > 0$ 使用 $h[0] = 0$（投机）

### 3.3 误差定义

对 rank $r > 0$，定义误差：

$$
\delta h[t] = h^{true}[t] - h^{partial}[t]
$$

初始误差：$\delta h[0] = h_{mid}$（从前一个 rank 收到的真实初始状态）

### 3.4 误差传播推导

**$v_{new}$ 的误差：**

$$
\delta v_{new}[t] = v_{new}^{true}[t] - v_{new}^{partial}[t]
$$

$$
= (v[t] - w[t] \cdot h^{true}[t]) - (v[t] - w[t] \cdot h^{partial}[t])
$$

$$
\boxed{\delta v_{new}[t] = -w[t] \cdot \delta h[t]}
$$

**$h$ 的误差传播：**

$$
\delta h[t+1] = h^{true}[t+1] - h^{partial}[t+1]
$$

$$
= e^{g^{last}_t} \cdot \delta h[t] + k[t]^\top \cdot G_t \cdot \delta v_{new}[t]
$$

$$
= e^{g^{last}_t} \cdot \delta h[t] - k[t]^\top \cdot G_t \cdot w[t] \cdot \delta h[t]
$$

$$
\boxed{\delta h[t+1] = M[t] \cdot \delta h[t]}
$$

其中 $M[t] = e^{g^{last}_t} \cdot I_K - k[t]^\top \cdot G_t \cdot w[t]$，与原始递推的转移矩阵相同。

**$o$ 的误差：**

$$
\delta o[t] = o^{true}[t] - o^{partial}[t]
$$

$$
= q[t] \cdot \delta h[t] \cdot e^{g[t]} \cdot s + A^{causal}_t \cdot \delta v_{new}[t] \cdot s
$$

$$
\boxed{\delta o[t] = q[t] \cdot \delta h[t] \cdot e^{g[t]} \cdot s - A^{causal}_t \cdot w[t] \cdot \delta h[t] \cdot s}
$$

### 3.5 关键性质

误差传播是线性的：$\delta h$ 的演化只依赖 $\delta h$ 本身和本地数据（$k, w, g$），不依赖 $v$ 或 $h^{partial}$。这意味着：

1. **$h^{partial}[t]$ 不需要重新计算**，直接加修正量即可
2. **$v_{new}^{partial}[t]$ 不需要重新计算**，直接减修正量即可
3. **$o^{partial}[t]$ 不需要重新计算**，直接加修正量即可
4. 修正只需要维护一个 $\delta h \in \mathbb{R}^{K \times V}$，逐 chunk 传播

---

## 4. Correction Kernel

收到真实初始状态 $h_{mid}$ 后，串行遍历本 rank 的每个 chunk：

```
输入: δh = h_mid

for t = 0, 1, ..., NT_local - 1:
    # (1) 修正 h
    h[t] ← h[t] + δh

    # (2) 计算中间量
    b_wdh = w[t] · δh                                    ∈ R^{BT×V}

    # (3) 修正 v_new
    v_new[t] ← v_new[t] - b_wdh

    # (4) 修正 o
    δo = q[t] · δh · exp(g[t]) · s  -  A_causal[t] · b_wdh · s
    o[t] ← o[t] + δo

    # (5) 传播 δh
    b_wdh_gated = G_t · b_wdh                            ∈ R^{BT×V}
    δh ← exp(g_last_t) · δh  -  k[t]^T · b_wdh_gated    ∈ R^{K×V}

输出: δh（本 rank 的 final state 修正量）
```

---

## 5. 跨 Rank 通信

以 4 卡为例：

```
Rank 0: 用真实 h0 → 得到正确的 h_final[0]
        send(h_final[0]) → Rank 1

Rank 1: recv(h_final[0]) 作为 δh[0]
        correction kernel → 得到 δh_final
        h_final[1] = h_final_partial[1] + δh_final
        send(h_final[1]) → Rank 2

Rank 2: recv(h_final[1]) 作为 δh[0]
        correction kernel → 得到 δh_final
        h_final[2] = h_final_partial[2] + δh_final
        send(h_final[2]) → Rank 3

Rank 3: recv(h_final[2]) 作为 δh[0]
        correction kernel → 得到 δh_final
        h_final[3] = h_final_partial[3] + δh_final
```

通信量：每次传递 $[N, H, K, V]$ 的 float32 tensor。

---

## 6. 时间线

```
Rank 0: [ Step1-4 ][ Step5(真实h0) ][ Step6 ][ send h_final ]
Rank 1: [ Step1-4 ][ Step5(h0=0)   ][ Step6 ]  recv → [ correction ][ send ]
Rank 2: [ Step1-4 ][ Step5(h0=0)   ][ Step6 ]          recv → [ correction ][ send ]
Rank 3: [ Step1-4 ][ Step5(h0=0)   ][ Step6 ]                  recv → [ correction ]
```

- Step 1-4：所有 rank 完全并行，无通信
- Step 5：所有 rank 同时执行（rank > 0 投机）
- Step 6：所有 rank 同时执行（用 partial 结果）
- Correction：顺序传递，每个 rank 收到后修正 o, h, v_new

Rank > 0 的 Step 5 + Step 6 与 Rank 0 的计算完全重叠，correction kernel 是唯一的额外串行开销。

---

## 7. 复杂度分析

### 单卡

Step 5 串行递推 $NT$ 个 chunk，每个 chunk 主要计算：
- $w \cdot h$：$O(BT \cdot K \cdot V)$
- $k^\top \cdot v_{new}$：$O(K \cdot BT \cdot V)$

总计：$O(NT \cdot BT \cdot K \cdot V)$

### CP 并行（P 个 rank）

- 每个 rank 的 Step 5：$O(NT/P \cdot BT \cdot K \cdot V)$
- Correction kernel 每个 chunk：$O(BT \cdot K \cdot V)$（$w \cdot \delta h$）+ $O(BT^2 \cdot V)$（$A_{causal} \cdot \delta v_{new}$）+ $O(BT \cdot K \cdot V)$（$q \cdot \delta h$）
- Correction 总计：$O(NT/P \cdot (BT \cdot K \cdot V + BT^2 \cdot V))$

Correction 的计算量与 Step 5 + Step 6 的量级相当，但由于 rank > 0 的 Step 5 和 Step 6 已经与 rank 0 重叠执行，correction 是"额外"开销。当 $P$ 较小且 $NT$ 较大时，$NT/P$ 的缩减带来的收益大于 correction 的开销。

### 通信量

每次 rank 间传递：$N \cdot H \cdot K \cdot V \cdot 4$ bytes（float32）

例如 $N=1, H=32, K=128, V=128$：$1 \times 32 \times 128 \times 128 \times 4 = 2$ MB，远小于序列数据本身。
