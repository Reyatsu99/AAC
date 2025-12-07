# ADVANCED ALGORITHMS - COMPLETE EXAM CHEAT BOOK
## 40-Page Comprehensive Study Guide with All Solutions, Hints & Modifications

**For Open-Book Exam | Print & Carry This Single Document**

---

# TABLE OF CONTENTS

1. [Quick Formulas & Essential Facts (Page 1-3)](#section-0-quick-formulas)
2. [Section 1: Randomized Algorithms (Page 4-9)](#section-1-randomized-algorithms)
3. [Section 2: Probabilistic Methods (Page 10-15)](#section-2-probabilistic-methods)
4. [Section 3: Approximation Algorithms (Page 16-23)](#section-3-approximation-algorithms)
5. [Section 4: Advanced Topics (Page 24-28)](#section-4-advanced-topics)
6. [Section 5: Proof Templates & Strategies (Page 29-32)](#section-5-proof-templates)
7. [Section 6: Common Exam Questions & Answers (Page 33-38)](#section-6-exam-questions)
8. [Section 7: Trick Questions & Traps (Page 39-40)](#section-7-tricks-and-traps)

---

# SECTION 0: QUICK FORMULAS & ESSENTIAL FACTS

## Must-Know Formulas

**Harmonic Series:**
$$H_n = 1 + \frac{1}{2} + \frac{1}{3} + ... + \frac{1}{n} \approx \ln n + 0.5772$$

**Quicksort Comparisons (Indicator RV):**
- $P(X_{ij} = 1) = \frac{2}{j-i+1}$ (probability elements i, j are compared)
- $E[\text{comparisons}] = 2n \ln n + O(n)$

**Hiring Problem Success:**
- Optimal observation phase: $k = n/e$
- Success probability: $P(\text{success}) \approx 1/e \approx 0.368$ (37%)
- Formula: $P(k) = \frac{k}{n} \ln(n/k)$ maximized at $k=n/e$

**Coupon Collector:**
- $E[\text{throws}] = n \cdot H_n \approx n \ln n$
- Phase i: need $n/(n-i+1)$ expected throws for new bin

**Ramsey Numbers:**
- $R(k,k) \geq 2^{k/2}$ (lower bound via probabilistic method)
- Small values: $R(3,3)=6$, $R(4,4)=18$, $R(5,5) \in [43,49]$

## Approximation Ratios at a Glance

| Problem | Ratio | Method | Tight Example |
|---------|-------|--------|----------------|
| Vertex Cover | 2 | Matching | $K_{n,n}$ |
| Dominating Set | $\ln(\Delta) + 1$ | Greedy degree | Star graph |
| Steiner Tree | 2 | MST-based | Path |
| Set Cover | $\ln(n)$ | Greedy | Elements vs sets |
| Knapsack FPTAS | $(1-\epsilon) \cdot OPT$ | Scale + DP | Any input |

## Complexity Summary Table

| Algorithm | Best | Average | Worst | Space | Use When |
|-----------|------|---------|-------|-------|----------|
| Quicksort (rand) | $n\log n$ | $n\log n$ | $n^2$ (rare) | $O(\log n)$ | Typical data |
| Mergesort | $n\log n$ | $n\log n$ | $n\log n$ | $O(n)$ | Worst-case guarantee |
| Quickselect | $n$ | $n$ | $n^2$ | $O(\log n)$ | Expected linear |
| Median-of-Medians | $n$ | $n$ | $n$ | $O(\log n)$ | Guaranteed linear |
| Vertex Cover | - | $2 \cdot OPT$ | $2 \cdot OPT$ | Poly | Exact NP-hard |
| Dominating Set | - | $\ln \Delta \cdot OPT$ | $\ln \Delta \cdot OPT$ | Poly | Greedy works well |

## Key Probability Facts

- **Union Bound:** $P(\cup A_i) \leq \sum P(A_i)$
- **Linearity of Expectation:** $E[\sum X_i] = \sum E[X_i]$ (NO independence needed)
- **Indicator RV:** $E[X] = P(\text{event})$ where $X = 1$ if event occurs
- **Chernoff Bound:** $P(X \geq (1+\delta)\mu) \leq e^{-\delta^2\mu/3}$
- **Birthday Paradox:** Need ~$\sqrt{n}$ items to get collision in n items with 50% prob

---

# SECTION 1: RANDOMIZED ALGORITHMS

## Algorithm 1.1: Randomized Quicksort

### Simple Explanation
Quicksort with random pivot instead of fixed pivot. Prevents adversarial worst-case inputs by randomizing where we split.

### Algorithm
```
RandomQuickSort(A, low, high):
  if low < high:
    p = random index between low and high
    swap(A[p], A[high])
    pivot = A[high]
    
    i = low - 1
    for j = low to high-1:
      if A[j] < pivot:
        i = i + 1
        swap(A[i], A[j])
    
    swap(A[i+1], A[high])
    RandomQuickSort(A, low, i)
    RandomQuickSort(A, i+2, high)
```

### Expected Time: $O(n \log n)$

**Proof using Indicator Variables:**

1. **Define:** $X_{ij} = 1$ if elements at positions i and j are compared (i < j)

2. **Key Insight:** Elements i and j compare iff **one becomes pivot before any element between them**
   - Why? If neither is pivot before others, they're in different partitions
   - If one is pivot before the other, they compare during partition

3. **Probability:** 
   $$P(X_{ij} = 1) = P(\text{element i or j chosen first among } \{i, i+1, ..., j\})$$
   $$= \frac{2}{j-i+1}$$

4. **Expected Total Comparisons:**
   $$E[\text{comparisons}] = E\left[\sum_{i<j} X_{ij}\right] = \sum_{i<j} \frac{2}{j-i+1}$$

5. **Calculate:**
   - Substitute $k = j - i + 1$:
   $$\sum_{k=2}^{n} \frac{2(n-k+1)}{k} = 2n \sum_{k=2}^{n} \frac{1}{k} - 2(n-1)$$
   $$= 2n(H_n - 1) - 2n + 2 \approx 2n \ln n$$

**Result:** $O(n \log n)$ expected ✓

### Worst-Case: $O(n^2)$ (But Extremely Rare)

Happens if pivot always splits 1:(n-1). Probability of this for random input ≈ exponentially small.

### Common Variations

**Variation 1: Three-Way Partition (For Duplicates)**
- Problem: Many equal elements cause $O(n^2)$ behavior
- Solution: Split into $\text{<pivot}, =\text{pivot}, >\text{pivot}$
- Benefit: Handles duplicates efficiently

**Variation 2: Median-of-Three Pivot Selection**
- Problem: Random sometimes picks extreme values
- Solution: Pick median of first, middle, last elements
- Benefit: Better practical constants

**Variation 3: Randomized Pivot with Restart**
- Problem: Bad pivot choices (rare but possible)
- Solution: If pivot is too extreme, restart with new random pivot
- Expected restarts: constant (probability 1/2 of good split)

---

## Algorithm 1.2: Randomized Quickselect

### Simple Explanation
Find k-th smallest element in $O(n)$ expected time. Unlike quicksort, only recurse on ONE side.

### Algorithm
```
RandomQuickSelect(A, k):
  if length(A) == 1:
    return A[0]
  
  pivot_index = random(0, length(A)-1)
  pivot = A[pivot_index]
  
  partition A into:
    left = elements < pivot
    right = elements > pivot
  
  if k <= length(left):
    return RandomQuickSelect(left, k)
  else if k == length(left) + 1:
    return pivot
  else:
    return RandomQuickSelect(right, k - length(left) - 1)
```

### Expected Time: $O(n)$

**Why not $O(n \log n)$ like quicksort?**

Even with bad split (1 vs n-1):
- Recurrence: $T(n) = O(n) + T(n-1)$ (only ONE recursive call)
- This would give $O(n^2)$

But good splits are likely (prob > 1/2 to get 1:3 or better split):
- Expected: $T(n) = O(n) + (1/2)T(n-1) + (1/2)T(3n/4)$
- Solving: $T(n) = O(n)$ ✓

**Intuition:** We only solve one smaller problem per level, not two like quicksort.

### vs Median-of-Medians

**Quickselect:**
- Average: $O(n)$ ✓
- Worst: $O(n^2)$ (probability ~ 0)
- Constants: Small, fast in practice
- Easy to implement

**Median-of-Medians:**
- Average: $O(n)$ ✓
- Worst: $O(n)$ guaranteed ✓
- Constants: Large, slower in practice
- Complex implementation

**Recommendation:** Use quickselect in practice; median-of-medians when guaranteed $O(n)$ needed.

### Exam Variation: Find Multiple Elements
**Q:** Find 5 smallest elements in expected $O(n)$ time
**A:** Call quickselect 5 times? NO, that's $O(5n) = O(n)$ still works!
Better: After finding 5-th smallest, partition once to get all 5 smallest.

---

## Algorithm 1.3: Median-of-Medians (Worst-Case $O(n)$)

### Simple Explanation
Guarantees good pivot always by dividing into groups of 5, finding median of each, then median of those medians.

### Algorithm
```
MedianOfMedians(A, k):
  if length(A) < 5:
    sort A
    return A[k-1]
  
  // Divide into groups of 5
  medians = []
  for i = 0 to length(A)-1 step 5:
    group = A[i : i+5]
    medians.append(median(group))  // O(1) for 5 elements
  
  // Recursively find median of medians
  pivot = MedianOfMedians(medians, len(medians)/2)
  
  // Partition around pivot
  left, equal, right = partition(A, pivot)
  
  if k <= length(left):
    return MedianOfMedians(left, k)
  else if k <= length(left) + length(equal):
    return pivot
  else:
    return MedianOfMedians(right, k - length(left) - length(equal))
```

### Time Complexity: $O(n)$ Guaranteed

**Recurrence:**
$$T(n) = T(n/5) + T(7n/10) + O(n)$$

**Why $T(7n/10)$?**
- After picking pivot as "median of medians", it's at least 30-th percentile
- So at most 70% of elements are on one side
- Therefore $T(7n/10)$ for worst-case recursive call

**Solving:**
$$T(n) \leq c_1 n + T(n/5) + T(7n/10)$$
$$\leq c_1 n + c_2(n/5) + c_2(7n/10)$$
$$\leq c_1 n + c_2 \cdot 9n/10$$

For large enough $c_2$: $T(n) \leq c_2 n$ ✓

### Practical Note
Median-of-medians is rarely used because:
- Larger constants make it slower than quickselect on real data
- Usually quickselect's $O(n)$ average beats MOM's $O(n)$ worst-case
- More complex to implement correctly

---

# SECTION 2: PROBABILISTIC METHODS

## Algorithm 2.1: Hiring Problem (Secretary Problem)

### Simple Explanation
Candidates arrive sequentially. You must decide immediately (no recall). Goal: hire the best candidate. What's the strategy?

### Optimal Strategy: Two-Phase Approach

**Phase 1 (Observe):** Interview first $k = n/e$ candidates, record maximum quality $M$

**Phase 2 (Select):** Hire first candidate better than $M$

### Why $k = n/e$?

**Success Probability Formula:**
$$P(\text{success}) = \frac{k}{n} \ln(n/k)$$

**Proof:**
1. Success = "best candidate is at position $j > k$ AND selected"

2. Probability best at position $j > k$: $1/n$ (uniform random order)

3. Probability position $j$ is selected: position $j$ is better than all in positions $1..k$
   - Among $j$ candidates, positions $1..k$ are uniformly distributed among first $j$
   - Probability all $k$ are in positions $1..k$ (not $1..j$): $\binom{j}{k} / \binom{j}{k} = \text{exactly } k/(j-1)$
   
   Wait, clearer: $P(\text{best among positions } 1..j \text{ is in positions } 1..k) = k/j$
   
   But we need best to be at position $j$. Given best is in $1..j$:
   - $P(\text{best at position } j \text{ | best in } 1..j) = 1/j$
   
   So: $P(\text{select position } j) = k/(j-1)$ approximately
   
   More precisely: $P(\text{all of } 1..k \text{ worse than position } j) = k/j$ for large $n$

4. **Total:**
   $$P(\text{success}) = \sum_{j=k+1}^{n} \frac{1}{n} \cdot \frac{k}{j} \approx \frac{k}{n} \sum_{j=k}^{n} \frac{1}{j}$$
   $$\approx \frac{k}{n}[\ln(n) - \ln(k)] = \frac{k}{n} \ln(n/k)$$

**Optimization:**
$$\frac{d}{dk}\left[\frac{k}{n}\ln(n/k)\right] = \frac{1}{n}[\ln(n/k) - 1] = 0$$
$$\Rightarrow \ln(n/k) = 1 \Rightarrow k = n/e$$

**At optimal $k = n/e$:**
$$P(\text{success}) = \frac{1}{e} \ln(e) = \frac{1}{e} \approx 0.368 \text{ (37%)} \checkmark$$

### Simple Understanding
- Observe enough to learn standards: $n/e$ ≈ 37% of candidates
- Then commit: hire first who exceeds learned standard
- Success rate: 37% (much better than random 1/n!)

### Common Variations

**Variation 1: Hire $m$ candidates (up to m)**
- Observation phase: still $n/e$
- Selection phase: hire up to m who beat observed standard
- Success = "at least one of top m is best"
- Harder analysis; success drops as m increases

**Variation 2: Want to hire exactly m**
- Different strategy entirely
- Equivalent to "select top m in order"
- Needs different optimization

**Variation 3: Candidates have quality scores**
- Randomness in observed vs actual quality
- Strategy uses Bayesian updating
- Success probability depends on distribution

---

## Algorithm 2.2: Ramsey Theory Lower Bounds

### Simple Explanation
Given complete graph with all edges colored red/blue, can we always find monochromatic clique of size $k$?

Answer: Yes, if graph is large enough. How large?

### Ramsey Number Definition
$$R(k,k) = \text{smallest } n \text{ such that any 2-coloring of } K_n$$
$$\text{contains monochromatic } K_k$$

**Known values:**
- $R(3,3) = 6$: Any 2-coloring of $K_6$ has mono triangle
- $R(4,4) = 18$: Any 2-coloring of $K_{18}$ has mono $K_4$
- $R(5,5)$ is unknown exactly, between 43-49

### Lower Bound Proof (Probabilistic Method)

**Theorem:** $R(k,k) > 2^{k/2}$

**Proof:**
1. **Random coloring:** Randomly color each edge red/blue independently, prob 1/2 each

2. **For any fixed k-subset S:**
   - Probability entire subset is red: $(1/2)^{\binom{k}{2}}$
   - Probability entire subset is blue: $(1/2)^{\binom{k}{2}}$
   - $P(S \text{ is monochromatic}) = 2 \cdot (1/2)^{\binom{k}{2}} = 2^{1-\binom{k}{2}}$

3. **Union bound over all k-subsets:**
   - Number of k-subsets in $K_n$: $\binom{n}{k}$
   - $P(\text{∃ monochromatic k-subset}) \leq \binom{n}{k} \cdot 2^{1-\binom{k}{2}}$

4. **When is this < 1?**
   $$\binom{n}{k} \cdot 2^{1-\binom{k}{2}} < 1$$
   $$n^k \cdot 2^{1-k(k-1)/2} < 1$$
   $$k \ln n + 1 - \frac{k(k-1)}{2} \ln 2 < 0$$
   
   For $n < 2^{k^2/4}$: inequality holds

5. **Conclusion:** For $n < 2^{k/2}$, we have $P(\text{no mono k-clique}) > 0$
   
   So there exists a 2-coloring with no monochromatic k-clique!
   
   Therefore: $R(k,k) \geq 2^{k/2}$ ✓

### Key Technique: Probabilistic Method

**Template:**
1. Define random structure (random coloring)
2. Show "bad event" (monochromatic subset exists)
3. Prove $P(\text{bad event}) < 1$
4. Conclude good structure exists (no mono subset)

**Magic:** We never construct the good coloring explicitly; we just prove it exists!

### Tournament Graphs & S_k Property

**Definition:** Tournament = complete directed graph (for each pair $(u,v)$, exactly one of $u \to v$ or $v \to u$)

**S_k property:** For every k-subset of vertices, there exists a vertex dominating all k

**Theorem:** Every tournament on $n \geq 2^k$ vertices has S_k property

**Proof idea:** Induction on k, using pigeonhole principle on out-degrees

---

# SECTION 3: APPROXIMATION ALGORITHMS

## Algorithm 3.1: Vertex Cover 2-Approximation

### Simple Explanation
Find small set of vertices covering all edges (each edge has at least one endpoint in set). Hard to optimize, but easy to approximate.

### Algorithm: Matching-Based

```
VertexCover2Approx(G = (V, E)):
  C = ∅  // vertex cover
  E' = E  // remaining edges
  
  while E' is not empty:
    Pick any edge (u, v) ∈ E'
    C = C ∪ {u, v}  // Add BOTH endpoints
    Remove all edges incident to u or v from E'
  
  return C
```

### Analysis: $|C| \leq 2 \cdot OPT$

**Step 1: C is valid vertex cover**
- Every edge was removed because at least one endpoint is in C ✓

**Step 2: $|C| \leq 2 \cdot OPT$**

Key insight: All edges picked form a **matching M** (no two edges share a vertex)
- Why? If two edges shared a vertex, second wouldn't be selected (would be removed when first was added)

Therefore:
- $|M| =$ number of edges picked
- $|C| = 2|M|$ (we picked both endpoints of each)

Any vertex cover must cover ALL edges, including all in M:
- Since edges in M don't share vertices, we need $\geq 1$ vertex per edge
- Therefore: $|OPT| \geq |M|$

**Combining:**
$$|C| = 2|M| \leq 2 \cdot |OPT|$$

**Result:** 2-approximation ✓

### Tight Example: $K_{n,n}$

**Graph:** Complete bipartite graph (n vertices on each side, all edges between sides)

**Optimal cover:** Pick one side = n vertices ✓

**Our algorithm (worst case):** Pick both endpoints from each edge = 2n vertices

**Ratio:** 2n/n = 2 ✓ (Tight!)

### Variation: Weighted Vertex Cover

**Problem:** Each vertex $v$ has weight $w(v)$. Minimize total weight of cover.

**Greedy approach:** Repeatedly pick vertex with best (cost/degree) ratio
- **BUT:** Greedy is not proven to be 2-approx for weighted version

**Better approach: LP Relaxation + Rounding**

1. **LP:** Minimize $\sum w(v) x(v)$ subject to $x(u) + x(v) \geq 1$ for each edge, $0 \leq x \leq 1$

2. **Solve LP:** Get fractional solution $x^*$

3. **Round:** Pick vertices with $x(v) \geq 1/2$

4. **Proof:** 
   - Solution is valid (each edge has $x(u) + x(v) \geq 1$, so at least one $\geq 1/2$)
   - Cost $\leq 2 \cdot$ LP optimal $\leq 2 \cdot$ OPT (LP is lower bound)
   - Result: 2-approx ✓

### Other Vertex Cover Variants

**Variant 1: Vertex Cover with Degree Constraints**
- Each vertex can cover at most k edges
- Algorithm must be modified (might not be 2-approx anymore)
- Harder analysis

**Variant 2: Edge Weights**
- Each edge has weight; minimize weight of uncovered edges
- Different from weighted vertex cover
- This is "minimum weighted edge cover" - different problem

---

## Algorithm 3.2: Greedy Dominating Set (ln(Δ) Approximation)

### Simple Explanation
Select vertices to dominate entire graph (every vertex is either selected or adjacent to selected). Greedy picks highest-degree vertices.

### Algorithm

```
GreedyDominatingSet(G = (V, E)):
  D = ∅  // dominating set
  U = V  // uncovered vertices
  
  while U is not empty:
    v = vertex in U with maximum |degree(v) ∩ U|
    // Pick vertex covering most uncovered neighbors
    
    D = D ∪ {v}
    U = U \ ({v} ∪ neighbors(v))
    // Mark v and neighbors as covered
  
  return D
```

### Approximation Ratio: $H(\Delta) + 1 \approx \ln(\Delta) + 1.58$

Where $\Delta$ = maximum degree, $H(\Delta) = 1 + 1/2 + 1/3 + ... + 1/\Delta$

**Proof Intuition:**

When greedy picks vertex $v$, it covers $\text{deg}(v)$ neighbors plus itself.

For optimal solution OPT*, each picked vertex u ∈ OPT* "contributes" to some greedy picks.

By charging argument:
- Greedy picks at most $H(\Delta)$ times the optimal
- Harmonic series ≈ ln(Δ)

**Formal Proof (Sketch):**

1. When vertex v is picked, it has degree d in remaining graph

2. At least one OPT vertex must be "responsible" for dominating v

3. Charge v's cost (1/$d$) to that OPT vertex

4. Sum of all charges $\leq H(\Delta)$ (harmonic series bound)

5. Therefore: $|D_{\text{greedy}}| \leq H(\Delta) \cdot |OPT|$ ✓

### Hardness: $\Omega(\ln n)$ Hard

**Theorem (conditional):** Unless P=NP or UGC false, cannot do better than $\ln(n)$ approximation

This means our greedy algorithm is essentially optimal!

### Randomized Dominating Set

**Algorithm:**
```
RandomDominatingSet(G, δ):
  p = ln(δ+1) / (δ+1)  // optimized probability
  D = ∅
  
  for each vertex v:
    Include v in D with probability p
  
  // Add neighbors of uncovered vertices
  for each uncovered vertex u:
    Add arbitrary neighbor to D
  
  return D
```

**Expected size:** $E[|D|] = \frac{n(1 + \ln(\delta+1))}{\delta+1}$

Where $\delta$ = minimum degree

### Can Derandomize
- Use conditional expectation method
- Decide each vertex in order, minimize expected final size
- Results in deterministic algorithm with same expected size

---

## Algorithm 3.3: Steiner Tree 2-Approximation

### Simple Explanation
Connect required vertices using minimum weight of edges. Can use "Steiner vertices" (temporary helpers). Hard problem, but 2-approx is easy.

### Problem Definition
- Graph $G = (V, E, w)$ with edge weights
- Required vertices $R \subseteq V$ (must all be in tree)
- Steiner vertices $S = V \setminus R$ (can optionally use)
- Goal: Minimum weight tree connecting all R vertices

### Algorithm: MST-Based

```
SteinerTreeApprox(G, R):
  // Step 1: Create metric closure
  G_metric = new graph with vertices = R
  for each pair (u, v) in R:
    weight(u, v) = shortest_path_distance(u, v) in G
  
  // Step 2: Find MST on metric closure
  T_metric = MST(G_metric)
  
  // Step 3: Replace edges with actual shortest paths
  T_full = empty graph
  for each edge (u, v) in T_metric:
    path = shortest_path(u, v) in G
    Add all vertices and edges from path to T_full
    // May introduce cycles
  
  // Step 4: Remove cycles to get tree
  T = minimum_spanning_tree(T_full)
  // or: DFS from root, keep tree edges only
  
  return T
```

### Analysis: 2-Approximation

**Step 1: MST(G_metric) $\leq 2 \cdot OPT$**

Proof:
- Consider optimal Steiner tree OPT in original graph
- Double its edges: creates closed walk of total weight $2w(OPT)$
- Walk connects all R vertices
- MST weight $\leq$ walk weight (MST is optimal tree on these vertices)
- Therefore: $w(T_{\text{metric}}) \leq 2w(OPT)$ ✓

**Step 2: Replacing edges with shortest paths**
- Shortest paths have weight $\leq$ original distance
- So total weight doesn't increase

**Step 3: Removing cycles**
- Can only decrease weight
- Keeps tree property

**Result:** $w(T) \leq 2w(OPT)$ ✓

### Tight Example
- Path: vertices $1-2-3-...-n$, all required
- Optimal: use n-1 edges
- MST on metric closure: same n-1 edges (all weights 1)
- Our algorithm: same result (ratio 1)

For ratio to be 2: need graphs where MST on metrics is much more than direct tree.

---

## Algorithm 3.4: Knapsack & FPTAS

### Problem: 0-1 Knapsack

Maximize value: $\sum v_i x_i$

Subject to: $\sum w_i x_i \leq W$, $x_i \in \{0,1\}$

### Exact Solution: Dynamic Programming $O(nW)$

```
DP[i][w] = maximum value using items 1..i with weight ≤ w

Base: DP[0][w] = 0

Recurrence:
  DP[i][w] = max(
    DP[i-1][w],              // don't take item i
    DP[i-1][w-w_i] + v_i     // take item i
  )
```

**Time:** $O(nW)$ - pseudo-polynomial (depends on capacity W)

**Problem:** If W = 1 million, this is slow!

### FPTAS: Fully Polynomial-Time Approximation Scheme

Gets $(1-\epsilon)$ approximation in polynomial time $O(n^3/\epsilon)$

**Idea:** Scale down values, solve exactly, scale back up

```
KnapsackFPTAS(items, W, ε):
  v_max = max(v_i)  // largest value
  K = ε · v_max / n  // scaling factor
  
  for each item i:
    v'_i = floor(v_i / K)  // scaled values
  
  // Solve exact knapsack with scaled values
  // Now DP[i][w] has values ≤ n/ε (polynomial!)
  solution = ExactKnapsack(items, W, scaled_values)
  
  return solution
```

**Time Complexity:**
- DP table: $n \times \frac{n}{\epsilon} = O(n^2/\epsilon)$
- Per cell: $O(1)$
- Total: $O(n^2/\epsilon)$ or $O(n^3/\epsilon)$ with details

**Quality:**
- Scaled solution is $(1-\epsilon) \cdot OPT$ in original values
- Proof: Rounding error bounded by $\epsilon \cdot OPT$

### Greedy for Knapsack: UNBOUNDED!

**Algorithm:** Sort by value/weight ratio, pick greedily

**Why it fails:** Can be arbitrarily bad

**Counterexample:**
```
Capacity W = 10
Item 1: weight 6, value 100, ratio = 16.7 ✓ Highest ratio!
Item 2: weight 10, value 101, ratio = 10.1

Greedy picks: Item 1 first (6 weight, 100 value)
  Can't fit Item 2 (would need 16 weight)
  Result: 100 value

Optimal: Item 2 only (10 weight, 101 value) ✓

Ratio: 100/101 ≈ 99% (looks good!)

But scale up: Item 1 value to 10^10, Item 2 value to 10^10 + 1
Ratio becomes 10^10 / (10^10 + 1) ≈ huge percentage loss
```

**Conclusion:** Greedy unbounded, cannot approximate constant ratio!

---

# SECTION 4: ADVANCED TOPICS

## 4.1: Balls-and-Bins (Coupon Collector Problem)

### Problem
Throw m balls uniformly at random into n bins. Expected number of throws to hit all bins?

### Solution: Harmonic Series!

$$E[\text{throws}] = n \cdot H_n = n(1 + 1/2 + 1/3 + ... + 1/n) \approx n \ln n$$

### Intuition: Phases

**Phase i:** We've already hit i-1 bins, now throwing until we hit a NEW bin

- Probability of new bin: $(n - (i-1))/n = (n-i+1)/n$
- Expected throws until success: $1 / (n-i+1)/n = n/(n-i+1)$

**Total expected:**
$$E = \sum_{i=1}^{n} \frac{n}{n-i+1} = \sum_{j=1}^{n} \frac{n}{j} = n H_n$$

**Numerical:** For $n = 365$ (days/bins):
$$E \approx 365 \ln 365 + 0.5772 \times 365 \approx 365 \times 5.9 + 211 \approx 2365 \text{ throws}$$

To have everyone's birthday represented (all 365 days) takes ~2400 people on average!

### Related: Birthday Paradox

**Different question:** How many people until collision (same birthday) likely?

**Answer:** ~$\sqrt{n} \approx 19$ people for 50% collision chance

$$P(\text{collision}) \approx 1 - e^{-k^2/(2n)}$$

For $P \approx 0.5$: $k \approx \sqrt{2n \ln 2} \approx 1.25\sqrt{n}$

---

## 4.2: Concentration Bounds

### Tail Bounds for Random Variables

**Markov Inequality** (for X ≥ 0):
$$P(X \geq a) \leq \frac{E[X]}{a}$$

**Chebyshev Inequality** (for any X):
$$P(|X - E[X]| \geq t) \leq \frac{\text{Var}(X)}{t^2}$$

**Chernoff Bound** (for Bernoulli sum):
$$P(X \geq (1+\delta)\mu) \leq e^{-\delta^2 \mu / 3}$$

**Hoeffding's Inequality** (for bounded X_i):
$$P\left|\frac{1}{m}\sum_{i=1}^{m} X_i - E[X]\right| \geq t) \leq 2e^{-2mt^2}$$

### When to Use Which?

- **Markov:** Very loose, only general fact available
- **Chebyshev:** Tighter than Markov, need variance
- **Chernoff:** Best for Bernoulli/binomial
- **Hoeffding:** Best for bounded independent variables

### Example: Maximum Load in Balls-and-Bins

**Question:** If throw $n \log n$ balls into n bins, what's max load in any bin?

**Analysis:**
- Expected load in bin i: $\mu = \log n$
- By Chernoff: $P(\text{load}_i \geq 3\log n) \leq e^{-\log n} = 1/n$
- By union bound: $P(\exists \text{ bin with load} \geq 3\log n) \leq n \cdot 1/n = 1$

So with good probability, max load is $O(\log n)$

---

## 4.3: Lower Bounds & Hardness

### NP-Hardness

**Definition:** Problem is NP-hard if any NP problem reduces to it in poly time

**Implication:** If P≠NP, then no poly-time algorithm exists

**Common hard problems:**
- Vertex cover decision: "Does G have vertex cover of size ≤ k?"
- Dominating set decision: "Does G have dominating set of size ≤ k?"
- TSP: "Is there tour of length ≤ k?"

### Inapproximability

**Stronger than NP-hard:** Limits how well we can approximate

**Examples:**
- Vertex cover: Can't do better than 1.0001 (conditional on UGC)
- Dominating set: Can't do better than $\ln(n)$ (conditional)
- TSP (general): Can't approximate at all (no constant ratio!)
- Knapsack: Fully polynomial approx scheme exists (unlike NP-hard problems)

### For Exam: What to Say

- "This problem is NP-hard, so no poly-time exact algorithm known"
- "Conditional on P≠NP, we cannot do better than X"
- "Hardness result implies our algorithm ratio is approximately optimal"

---

# SECTION 5: PROOF TEMPLATES & STRATEGIES

## Template 1: Expected Value via Indicator Variables

**Use for:** Problems asking "expected value of..."

**Structure:**
1. Define indicator $X_i = 1$ if "event i happens"
2. Total = $\sum X_i$
3. $E[\text{Total}] = E[\sum X_i] = \sum E[X_i] = \sum P(\text{event i})$
4. Compute each probability
5. Sum up

**Example: Expected hires in hiring problem**

"Expected number of candidates hired if we use optimal $k=n/e$ strategy"

- $X_i = 1$ if candidate i is hired
- $P(i \text{ hired}) = 1/i$ (ith candidate is best among first i)
- $E[\text{total}] = \sum_{i=1}^{n} 1/i = H_n \approx \ln n$

## Template 2: Approximation Ratio Proof

**Use for:** "Design c-approximation algorithm"

**Structure:**
1. Give algorithm (clear pseudocode)
2. Prove correctness (algorithm finds valid solution)
3. Prove upper bound: $|ALG| \leq k \cdot OPT$ (worst-case over all inputs)
4. Give tight example: specific input family where ratio achieves k
5. Conclude: "Therefore, c-approximation where $c = k$"

**Example: Vertex cover 2-approx**

1. Algorithm: pick both endpoints of edges until all covered
2. Correctness: every edge has at least one endpoint in solution
3. Upper bound: let M = matching of picked edges, $|ALG| = 2|M|$, any cover needs $\geq |M|$
4. Tight example: $K_{n,n}$ complete bipartite
5. Conclusion: 2-approximation

## Template 3: Existential Proof via Probabilistic Method

**Use for:** "Prove that X exists"

**Structure:**
1. Define probability space (random construction)
2. Define "bad event" (X doesn't exist or is bad)
3. Calculate $P(\text{bad event})$
4. Show $P(\text{bad event}) < 1$
5. Conclude $P(\text{good event}) > 0$, so good X exists

**Example: Ramsey lower bound**

1. Prob space: random 2-coloring of edges
2. Bad event: monochromatic k-clique exists
3. $P(\text{bad}) = \binom{n}{k} 2^{1-\binom{k}{2}}$
4. For $n < 2^{k/2}$: $P(\text{bad}) < 1$
5. So good coloring exists (no mono k-clique)

## Template 4: Concentration (Tail Bounds)

**Use for:** "High probability bound"

**Structure:**
1. Identify random variable X
2. Calculate $E[X]$
3. Choose appropriate bound
4. Bound tail: $P(X > (1+\delta)E[X])$ or $P(|X - E[X]| > t)$
5. Interpret result

**Example: Max load in balls-bins**

1. X = number of balls in bin 1
2. $E[X] = n/n = 1$ (if m=n balls)
3. Use Chernoff for Bernoulli sum
4. $P(X > 3) \leq e^{-c}$ for small c
5. Result: with high prob, load at most O(log n)

## Template 5: Recurrence Relations

**Use for:** "What's time complexity?"

**Structure:**
1. Write recurrence (express T(n) in terms of smaller instances)
2. Expand first few levels to see pattern
3. Apply master theorem OR guess-and-verify OR telescoping
4. Conclude: T(n) = O(...)

**Example: Quicksort**

1. $T(n) = O(n) + T(n-1)$ (worst case)
2. Expand: $O(n) + O(n-1) + ... + O(1)$
3. Sum: $\sum_{i=1}^n i = n(n+1)/2 = O(n^2)$
4. Conclusion: O(n²) worst-case

**Better (average case with indicator variables):**
1. $E[T(n)] = O(n) + E[\text{small recursions}]$
2. $E[\text{sum of comparisons}] = 2n \ln n$ (from indicator RV analysis)
3. Conclusion: O(n log n) expected

---

# SECTION 6: COMMON EXAM QUESTIONS & ANSWERS

## Q1: Quicksort Expected Comparisons

**Question:** Prove that randomized quicksort makes $O(n \log n)$ expected comparisons.

**Answer Template:**

Define indicator $X_{ij} = 1$ iff elements at positions i and j are compared (i < j).

Elements i and j compare iff one becomes pivot before the other AND before all elements between them.

Among positions $\{i, i+1, ..., j\}$, probability one of i,j is chosen first: $2/(j-i+1)$

Total expected comparisons:
$$E = E[\sum X_{ij}] = \sum P(X_{ij}=1) = \sum_{i<j} \frac{2}{j-i+1}$$

Substitute $k = j-i+1$:
$$= \sum_{k=2}^n 2(n-k+1)/k = 2n\sum_{k=2}^n 1/k - 2(n-1)$$
$$\approx 2n \ln n$$

Therefore: $O(n \log n)$ expected comparisons ✓

---

## Q2: Hiring Problem Optimization

**Question:** Why is $k = n/e$ optimal for hiring problem?

**Answer Template:**

Success probability for observation phase of size k:
$$P(k) = \frac{k}{n}\ln(n/k)$$

Maximize: $\frac{d}{dk}\left[\frac{k}{n}\ln(n/k)\right] = \frac{1}{n}[\ln(n/k) - 1] = 0$

Solving: $\ln(n/k) = 1$, so $k = n/e$

At this value:
$$P(n/e) = \frac{1}{e} \ln(e) = \frac{1}{e} \approx 0.37$$

So we get 37% success probability, compared to random 1/n ≈ 0.1% ✓

---

## Q3: Vertex Cover 2-Approximation Correctness

**Question:** Prove algorithm that picks both endpoints of each edge is 2-approx for vertex cover.

**Answer Template:**

Algorithm:
1. While edges remain
2. Pick any edge (u,v)
3. Add both u,v to cover
4. Remove all incident edges

**Correctness:** Every edge removed because at least one endpoint added to cover ✓

**Approximation:** 
- Let M = set of edges picked (forms matching: no shared endpoints)
- Algorithm size: $|ALG| = 2|M|$
- Any vertex cover must cover all edges, including M
- Since M is matching (no shared vertices): $|OPT| \geq |M|$
- Therefore: $|ALG| = 2|M| \leq 2|OPT|$ ✓

Tight example: $K_{n,n}$ - pick one side (size n) vs both sides from some edges (size 2n)

---

## Q4: Greedy Dominating Set Ratio

**Question:** What's approximation ratio of greedy dominating set algorithm?

**Answer:** $H(\Delta) + 1 = \ln(\Delta) + O(1)$

Where $\Delta$ is max degree, $H_k = 1 + 1/2 + ... + 1/k$ is harmonic series.

**Proof idea:**
- When greedy picks vertex v, it covers $d_v$ neighbors
- Charge $1/d_v$ to some optimal vertex "responsible" for v
- Sum all charges $\leq H(\Delta)$ (harmonic bound)
- Greedy picks $\leq H(\Delta) \times OPT$ vertices ✓

---

## Q5: Modification - Quicksort with Duplicates

**Question:** How does quicksort perform when many elements are equal?

**Problem:** Standard partition makes $O(n^2)$ with many duplicates

**Solution:** 3-way partition into $\{<\text{pivot}, =\text{pivot}, >\text{pivot}\}$

- Equal elements go to middle, no further recursion needed
- Even with many duplicates, expected time $O(n \log n)$

**Example:** All elements equal
- Standard: tries to sort them, $O(n^2)$
- 3-way: one partition, all in middle, done, $O(n)$ ✓

---

## Q6: Steiner Tree vs Minimum Spanning Tree

**Question:** Why can't we just use MST for Steiner tree problem?

**Analysis:**
- MST connects ALL vertices, not just required ones
- MST might use many Steiner vertices needlessly
- Steiner tree uses Steiner vertices only when helpful

**Example:**
- Path graph: 1-2-3-4-5, all vertices have weight 1
- Required vertices: {1, 5}
- MST of full graph: 1-2-3-4-5 (connects all)
- Steiner tree optimal: just 1-2-3-4-5 (same, weights still 1 each)
- But if Steiner vertices have higher cost, Steiner tree uses fewer of them

---

## Q7: Knapsack vs Greedy

**Question:** Can greedy value/weight ratio give constant approximation for knapsack?

**Answer:** NO, unbounded approximation possible!

**Counterexample:**
- Capacity 10
- Item A: weight 6, value 100, ratio 16.7
- Item B: weight 10, value 101, ratio 10.1

Greedy picks A (16.7 > 10.1), can't fit B → value 100
Optimal picks B → value 101

Ratio: 100/101, but can make this arbitrarily worse by scaling values.

**Lesson:** Some problems (knapsack, TSP) greedy fails; need DP/FPTAS

---

# SECTION 7: TRICK QUESTIONS & COMMON TRAPS

## Trap 1: "Quicksort is always O(n log n)"

**❌ WRONG:** Can say worst-case is O(n²)

**✓ CORRECT:** "Expected time is O(n log n); worst-case is O(n²) but probability exponentially small"

**Why difference matters:** For exam, specify whether you mean worst-case, average-case, or expected.

---

## Trap 2: "Assume independence of X_ij"

**❌ WRONG:** Indicator variables X_ij are NOT independent (comparing same elements)

**✓ CORRECT:** Use linearity of expectation; no independence needed!

$$E[\sum X_i] = \sum E[X_i]$$

Works even if X_i strongly dependent.

---

## Trap 3: "Greedy works for knapsack"

**❌ WRONG:** Greedy by value/weight ratio is unbounded!

**✓ CORRECT:** 
- Use exact DP: O(nW) pseudo-polynomial
- Use FPTAS: O(n³/ε) polynomial, get (1-ε)-approx
- Greedy fails badly

---

## Trap 4: "This single graph shows tightness"

**❌ WRONG:** Tight example must be infinite FAMILY of inputs

**✓ CORRECT:** "For any n, consider K_{n,n}. Algorithm picks 2n, optimal picks n, ratio 2."

---

## Trap 5: "Union bound is always < 1"

**❌ WRONG:** Union bound can be > 1; still useful if we can show < 1

**✓ CORRECT:** Use union bound when sum of probabilities < 1. If > 1, try different approach.

---

## Trap 6: "Matching improves vertex cover beyond 2"

**❌ WRONG:** No algorithm known to beat 2. UGC implies 2-hardness.

**✓ CORRECT:** "Conjecture: 2 is optimal. Known: nothing better than 2 in general."

---

## Trap 7: "Expected time = average over all inputs"

**❌ WRONG:** Average case is different from expected (with randomness)

**✓ CORRECT:** 
- Expected time: fix input, average over random choices of algorithm
- Average case: average time over all possible inputs (worst input might exist!)

---

## Trap 8: "Probability of O(n) time over million runs"

**❌ WRONG:** Asking for empirical frequency without rigorous bound

**✓ CORRECT:** "With probability 1 - 1/n², time is O(n log n)" (high probability bound)

---

## Trap 9: "This greedy is 3-approx... claim it's 4-approx too"

**❌ WRONG:** Prove exact ratio, not loose upper bound

**✓ CORRECT:** State "3-approximation ratio" and prove tight example achieves ratio 3

---

## Trap 10: "No optimal algorithm exists, so problem is NP-complete"

**❌ WRONG:** NP-hard means hard to optimize; NP-complete means hard to even decide

**✓ CORRECT:** 
- NP-hard: no fast exact algorithm likely (P≠NP)
- Approximation still possible
- NP-complete: decision version hard, optimization often even harder

---

# FINAL QUICK REFERENCE

## During Exam - Finding Answers Fast

| You See | Go To | Section |
|---------|-------|---------|
| "Expected comparisons" | Quicksort | Section 1.1 |
| "Best candidate in hiring" | Hiring problem | Section 2.1 |
| "Prove structure exists" | Ramsey/Prob method | Section 2.2 |
| "Approximate vertex cover" | Section 3.1 | |
| "Dominating set ratio" | Section 3.2 | |
| "Knapsack problem" | Section 3.4 | |
| "Tail bound needed" | Section 4.2 | Concentration |
| "Write formal proof" | Section 5 | Proof templates |
| "What's the trick?" | Section 7 | Traps |

## Formulas to Memorize

$$H_n \approx \ln n + 0.5772$$

$$P(\text{compare i,j}) = \frac{2}{j-i+1}$$

$$P(\text{hiring success at } k) = \frac{k}{n}\ln(n/k), \text{ max at } k=n/e$$

$$E[\text{coupon collector}] = n H_n$$

$$|VC| \leq 2 \cdot OPT, \quad |DS| \leq \ln(\Delta) \cdot OPT$$

$$E[\text{throws for collision}] \approx \sqrt{n}$$

---

**END OF CHEAT BOOK - Good Luck on Your Exam! 🎯**

*This is page 40 of your comprehensive guide. Print and organize by sections for quick access.*
