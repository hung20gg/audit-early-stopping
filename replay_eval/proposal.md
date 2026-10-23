\section{Evaluation Methodology}

Evaluating many early exit methods independently can be expensive, because each method may require a separate model rollout. To reduce this evaluation cost, we use a replay-based protocol. For each input example, we first generate one full reasoning trace from the model and store the generated tokens together with available metadata, such as token log probabilities, stopping position, reasoning length, and final-answer length. We then replay this token sequence and evaluate all early exit methods on the same observed prefix stream.

Let $\mathcal{D}$ be the evaluation dataset and let $M$ be the reasoning model. We denote the set of early exit methods by
\[
\mathcal{I} = \{1, 2, \ldots, n\}.
\]
Each method $i \in \mathcal{I}$ is defined by three components:
\[
i = (S_i, \pi_i, G_i),
\]
where $S_i$ is the checkpoint function, $\pi_i$ is the verifier, and $G_i$ is the stop rule. The checkpoint function $S_i$ decides whether the current prefix should be checked. The verifier $\pi_i$ estimates whether the current reasoning state is sufficient to answer. The stop rule $G_i$ converts the verifier output into a binary stop decision.

Since different methods may check different positions, we define a shared checkpoint function as the union of all method-specific checkpoint functions:
\[
S_{\cup}(r_{1:t}, z_t) =
\bigvee_{i \in \mathcal{I}} S_i(r_{1:t}, z_t),
\]
where $r_{1:t}$ is the current reasoning prefix and $z_t$ is the latest generated token. For example, if $S_1$ checks positions $\{1,3\}$ and $S_2$ checks positions $\{2,3\}$, then $S_{\cup}$ checks positions $\{1,2,3\}$. This allows all methods to be evaluated in one replay pass.

For each example $x \in \mathcal{D}$, we first generate a full reasoning trajectory:
\[
(r_{1:T}, y^{\mathrm{full}}) \sim M(x),
\]
where $r_{1:T}$ is the full reasoning trace and $y^{\mathrm{full}}$ is the final answer generated after natural stopping. During replay, each early exit method only observes the prefix $r_{1:t}$ at time $t$. Future tokens $r_{t+1:T}$ and the full answer $y^{\mathrm{full}}$ are not given to the verifier. They are used only for offline analysis and scoring.

When a method decides to stop at time $t$, we append the end-of-thinking token $\tau$, such as \texttt{</think>}, to the prefix and ask the model to generate the final answer:
\[
y_i \sim M(x, r_{1:t} \circ \tau).
\]
The output $y_i$ is then evaluated by the scoring function $V$, which may include accuracy, token usage, answer entropy, answer drift, and post-stop thinking leakage. After method $i$ stops, it is removed from the active method set for that example.

\begin{algorithm}[t]
\caption{Replay-Based Evaluation for Early Exit Methods}
\label{alg:replay_eval}
\begin{algorithmic}[1]
\REQUIRE Dataset $\mathcal{D}$, model $M$, method set $\mathcal{I}$, scorer $V$, end-of-thinking token $\tau$
\STATE Initialize result set $\mathcal{R} \leftarrow \emptyset$
\FOR{each example $(x, g) \in \mathcal{D}$}
    \STATE Generate full trajectory $(r_{1:T}, y^{\mathrm{full}}) \sim M(x)$
    \STATE Store full-generation metadata
    \STATE Initialize active method set $\mathcal{A} \leftarrow \mathcal{I}$
    \FOR{$t = 1$ to $T$}
        \STATE Let $z_t$ be the latest token in prefix $r_{1:t}$
        \STATE Find candidate methods:
        \[
        \mathcal{C}_t \leftarrow \{i \in \mathcal{A} \mid S_i(r_{1:t}, z_t) = 1\}
        \]
        \FOR{each method $i \in \mathcal{C}_t$}
            \STATE Compute verifier output $u_{i,t} \leftarrow \pi_i(x, r_{1:t})$
            \IF{$G_i(u_{i,t}, r_{1:t}) = 1$}
                \STATE Generate early-exit answer $y_i \sim M(x, r_{1:t} \circ \tau)$
                \STATE Compute score $s_i \leftarrow V(y_i, g, r_{1:t}, y^{\mathrm{full}})$
                \STATE Add $(x, i, t, y_i, s_i)$ to $\mathcal{R}$
                \STATE Remove method $i$ from active set $\mathcal{A}$
            \ENDIF
        \ENDFOR
        \IF{$\mathcal{A} = \emptyset$}
            \STATE \textbf{break}
        \ENDIF
    \ENDFOR
    \FOR{each remaining method $i \in \mathcal{A}$}
        \STATE Compute full-reasoning score $s_i \leftarrow V(y^{\mathrm{full}}, g, r_{1:T}, y^{\mathrm{full}})$
        \STATE Add $(x, i, T, y^{\mathrm{full}}, s_i)$ to $\mathcal{R}$
    \ENDFOR
\ENDFOR
\RETURN Result set $\mathcal{R}$
\end{algorithmic}
\end{algorithm}

This protocol has two advantages. First, all early exit methods are evaluated on the same reasoning trajectory, which makes their stopping positions directly comparable. Second, it avoids repeating the full reasoning rollout for every method. However, the final reported efficiency must still include the actual cost required by each method, including the reasoning prefix length, verifier calls, extra probing tokens, and final-answer tokens after early stopping. Therefore, the replay protocol reduces experimental cost, but it should not hide the true deployment cost of each early exit method.

\subsection{Continuation reasoning-error rate}

For a fixed reasoning budget \(t\), let \(C_t\) indicate that the answer
generated from reasoning prefix \(r_t\) is correct, and let \(C_T\) indicate
that the paired full answer from the same question and run is correct. We
define the \emph{continuation reasoning-error rate}, also called the
\emph{correct-answer degradation rate}, as
\[
q_t=P(C_T=0\mid C_t=1).
\]
Conditional on the answer generated from reasoning prefix \(r_t\) being
correct, continued reasoning produces an incorrect final answer with
probability \(q_t\). Therefore,
\[
A_t\le A_T+A_tq_t\le A_T+q_t.
\]
We estimate \(q_t\) by pairing each fixed-budget answer with its full answer
within the same run. For every benchmark/model stratum and budget, we report
the number correct at \(t\), the number correct at \(t\) but wrong at full
generation, \(\widehat q_t\), and a question-clustered 95\% confidence
interval. We additionally report a stratified-overall estimate and the maximum
over budgets with a simultaneous confidence bound. An upper confidence bound
of at most 2\% provides strong support, one above 2\% and at most 5\% provides
moderate support, and one above 5\% provides poor support.

This quantity concerns answer correctness and should not be described as a
hallucination rate. Moreover, the logs evaluate the generated answer, not the
logical truth of every statement in \(r_t\). We therefore say that ``the
answer generated from \(r_t\) is correct,'' not that ``\(r_t\) is true.''

\section{Reference Implementation}

The protocol above is realized by the \texttt{replay\_eval} Python package. Adding a new early exit method requires only writing (or reusing) an $S$, a $\pi$, or a $G$; the replay loop is method-agnostic.

\subsection{Class-to-algorithm mapping}

\begin{center}
\begin{tabular}{ll}
\hline
Algorithm element & Class / location \\
\hline
Method $i = (S_i, \pi_i, G_i)$ & \texttt{EarlyExitMethod} (\texttt{replay\_eval/method.py}) \\
Checkpoint function $S_i$ & \texttt{CheckpointFn} subclasses (\texttt{replay\_eval/checkpoints.py}) \\
Verifier $\pi_i$ & \texttt{Verifier} subclasses (\texttt{replay\_eval/verifiers.py}) \\
Stop rule $G_i$ & \texttt{StopRule} subclasses (\texttt{replay\_eval/stop\_rules.py}) \\
Shared checkpoint $S_{\cup}$ & Orchestrator loop in \texttt{ReplayEvaluatorAgent.\_replay} \\
Method set $\mathcal{I}$ & \texttt{ReplayEvaluatorAgent.methods} (built or spec-driven) \\
Active set $\mathcal{A}$ & Local \texttt{active} list mutated during replay \\
Full trajectory $(r_{1:T}, y^{\mathrm{full}})$ & \texttt{Response} returned by \texttt{\_obtain\_trajectory} \\
Prefix $r_{1:t}$ + $z_t$ & \texttt{VerifierContext.prefix} / \texttt{.latest\_token} \\
Early answer $y_i \sim M(x, r_{1:t}\circ\tau)$ & \texttt{\_generate\_early\_answer} (fresh model call) \\
Fallback answer for $i \in \mathcal{A}$ at $T$ & \texttt{\_extract\_full\_answer} \\
Result row $(x, i, t, y_i, s_i)$ & \texttt{MethodResult} (\texttt{replay\_eval/method.py}) \\
Result set $\mathcal{R}$ & List of \texttt{MethodResult}, exposed as \texttt{response.method\_results} \\
Scorer $V$ & Existing \texttt{evaluator} + \texttt{scripts/runner.py} serializer \\
\hline
\end{tabular}
\end{center}

\subsection{Component contracts}

\paragraph{\texttt{EarlyExitMethod}.} The composed triple. Constructed as
\[
\texttt{EarlyExitMethod}(name,\; S,\; \pi,\; G)
\]
and holds per-example mutable \texttt{state} that its $S$, $\pi$, and $G$ can share (e.g., last-check position, accumulated verifier cost).

\paragraph{\texttt{CheckpointFn} ($S_i$).} Callable \(\texttt{ctx} \mapsto \{\text{True}, \text{False}\}\). Given a \texttt{VerifierContext} exposing $r_{1:t}$, $z_t$, and $t$, it returns whether this position is a checkpoint. Built-in implementations:
\begin{itemize}
  \item \texttt{StopMarkerCheckpoint} --- fires when $r_{1:t}$ ends with any of a configured stop marker (e.g., DEER's chunk boundaries).
  \item \texttt{IntervalSignalCheckpoint} --- fires on a reflection signal but no more often than every \texttt{min\_interval} tokens (DTSR).
  \item \texttt{EveryNTokens} --- deterministic periodic checks.
  \item \texttt{AlwaysCheckpoint} --- $S \equiv 1$, used for tests / dense methods.
  \item \texttt{AnyCheckpoint} --- disjunction of children; the union $S_{\cup}$ is implemented implicitly by evaluating each method's $S$, but this class is available for composing per-method unions.
\end{itemize}

\paragraph{\texttt{Verifier} ($\pi_i$).} Callable \(\texttt{ctx} \mapsto u_{i,t}\). The verifier may make additional model calls through \texttt{ctx.agent} (e.g., \texttt{generate\_probe}, \texttt{sufficiency\_score}); those calls' prompt/completion tokens are tallied on the method's \texttt{state} and reported per method. Built-in implementations:
\begin{itemize}
  \item \texttt{ProbeLogprobVerifier} (DEER) --- returns the mean per-token probability over the next \texttt{probe\_tokens} tokens. Two modes:
    \begin{itemize}
      \item \texttt{source="probe"} --- issue a fresh $M(x, r_{1:t} \circ \tau)$ probe call and use its logprobs. Faithful to deployment cost.
      \item \texttt{source="trajectory"} --- read the next logprobs from the shared trajectory. Cheap replay approximation.
    \end{itemize}
  \item \texttt{LLMSufficiencyVerifier} (DTSR) --- prompts the model with a sufficiency template and extracts a score in $[0, 100]$ from \(\backslash\)\texttt{boxed\{\ldots\}}.
  \item \texttt{KeywordCountVerifier} --- counts keyword occurrences in $r_{1:t}$; useful for simple baselines.
\end{itemize}

\paragraph{\texttt{StopRule} ($G_i$).} Callable \((u_{i,t}, \texttt{ctx}) \mapsto \{\text{True}, \text{False}\}\). Built-ins:
\begin{itemize}
  \item \texttt{ThresholdGE(threshold)} --- stop iff $u \ge$ threshold.
  \item \texttt{ThresholdLE(threshold)} --- stop iff $u \le$ threshold (e.g., variance-based rules).
  \item \texttt{AlwaysStop} --- stop whenever $S$ triggers.
\end{itemize}

\paragraph{\texttt{VerifierContext}.} Bundle of information visible at time $t$: \texttt{question}, \texttt{prefix} ($r_{1:t}$ as a \texttt{Response} with logprobs), \texttt{latest\_token} ($z_t$), \texttt{position} ($t$), \texttt{agent}, runtime \texttt{kwargs}, and per-method \texttt{state}. It also carries an optional \texttt{trajectory} handle exclusively so that \emph{opt-in} replay approximations (e.g., \texttt{ProbeLogprobVerifier(source="trajectory")}) can peek at future logprobs. Faithful verifiers must ignore it.

\paragraph{\texttt{ReplayEvaluatorAgent}.} Implements Algorithm 1. Two flavors share the same replay core:
\begin{itemize}
  \item \texttt{ReplayEvaluatorAgent} (reuse) --- expects the trajectory to be supplied through \texttt{reuse\_generation\_response} (or \texttt{reuse\_generation\_content}); this fits the ``generate-once, replay-many'' workflow already used by \texttt{DTSRAgentReuseGeneration}.
  \item \texttt{ReplayEvaluatorAgentFullGeneration} --- rolls the trajectory itself before replaying.
\end{itemize}
When a method decides to stop, \texttt{\_generate\_early\_answer} bypasses any replay-based finalize path and calls the base LLM directly, so $y_i \sim M(x, r_{1:t} \circ \tau)$ is a real forward pass. Methods that never stop are attributed $y^{\mathrm{full}}$ via \texttt{\_extract\_full\_answer}, which reads $y^{\mathrm{full}}$ out of the trajectory when it contains \texttt{</think>} and otherwise runs a fresh finalize.

\paragraph{\texttt{MethodResult}.} One row of $\mathcal{R}$. Fields:
\[
\texttt{(method\_name, position, stopped\_early, verifier\_score, reasoning\_prefix, final\_answer, finish\_reason,}
\]
\[
\texttt{prompt\_tokens, completion\_tokens, verifier\_prompt\_tokens, verifier\_completion\_tokens)}.
\]
The verifier token counters make it possible to report the true deployment cost --- reasoning prefix length \emph{plus} verifier calls \emph{plus} extra probing tokens \emph{plus} final-answer tokens --- as the paragraph after Algorithm 1 requires.

\subsection{Adding a new method}

\begin{enumerate}
  \item Pick or write $S$ (subclass \texttt{CheckpointFn} or pass any callable).
  \item Pick or write $\pi$ (subclass \texttt{Verifier} or pass any callable).
  \item Pick or write $G$ (subclass \texttt{StopRule} or pass any callable).
  \item Compose: \texttt{EarlyExitMethod("my\_method", S, $\pi$, G)}.
  \item Optionally add a \texttt{build\_my\_method(...)} factory to \texttt{replay\_eval/methods.py} and reference it from \texttt{scripts/config.yml} as \texttt{\{factory: my.pkg.build\_my\_method, kwargs: \{\ldots\}\}}.
\end{enumerate}

\subsection{Logic diagram}

\begin{verbatim}
                       +---------------------------------------------+
                       | For each example (x, g) in dataset D        |
                       +---------------------------------------------+
                                             |
                                             v
                       +---------------------------------------------+
                       | _obtain_trajectory(question, kwargs)        |
                       |   * Full mode:  M(x)  -> (r_{1:T}, y_full)  |
                       |   * Reuse mode: supplied Response w/ logits |
                       +---------------------------------------------+
                                             |
                                             v
                       +---------------------------------------------+
                       | active <- copy(methods)                     |
                       | results <- []                               |
                       +---------------------------------------------+
                                             |
                                             v
                    +-----------------------------------------------+
                    | for t = 1..T over tokens of r_{1:T}           |
                    |   (loop exits once active is empty or t == T) |
                    +-----------------------------------------------+
                                             |
                                             v
                    +-----------------------------------------------+
                    | Build VerifierContext(prefix=r_{1:t}, z_t, t) |
                    +-----------------------------------------------+
                                             |
                                             v
                    +-----------------------------------------------+
                    | for each method i in active:                  |
                    |     if not S_i(ctx):    continue              |  <-- checkpoint (S)
                    |     u_i <- pi_i(ctx)                          |  <-- verifier    (pi)
                    |     if not G_i(u_i, ctx): continue            |  <-- stop rule   (G)
                    |     y_i <- _generate_early_answer(prefix)     |  fresh M(x, r_{1:t} o tau)
                    |     results.append(MethodResult(...))         |
                    |     active.remove(i)                          |
                    +-----------------------------------------------+
                                             |
                                             v
                    +-----------------------------------------------+
                    | if active is empty: break                     |
                    +-----------------------------------------------+
                                             |
                                             v
                    +-----------------------------------------------+
                    | for each remaining method i in active:        |
                    |     results.append(MethodResult using y_full) |
                    +-----------------------------------------------+
                                             |
                                             v
                       +---------------------------------------------+
                       | attach results to response, return R        |
                       +---------------------------------------------+
\end{verbatim}

The horizontal arrow flow, per method $i$ at a checkpoint, is exactly
\[
r_{1:t}, z_t \;\xrightarrow{S_i}\; \text{check?} \;\xrightarrow{\pi_i}\; u_{i,t} \;\xrightarrow{G_i}\; \text{stop?} \;\xrightarrow{M(x,\,r_{1:t}\circ\tau)}\; y_i.
\]
