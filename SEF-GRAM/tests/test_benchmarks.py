import torch

from experiments.ablation_suite import (
    AblationConfig,
    NeuralOnlyRetrievalModel,
    evaluate_neural_retrieval,
    mean_metrics,
    neural_retrieval_train_step,
)
from experiments.full_nqueens_benchmark import (
    NQueensRunConfig,
    batch_from_solutions,
    evaluate as evaluate_nqueens,
    solve_nqueens,
    solution_is_valid,
    teacher_forced_policy_loss,
)
from experiments.terminal_echo_benchmark import (
    EchoTerminalModel,
    TerminalEchoConfig,
    TinyCharTokenizer,
    make_terminal_task,
    train_step,
)
from experiments.terminal_retrieval_benchmark import (
    RetrievalConfig,
    RetrievalMemoryModel,
    TinyCharTokenizer as RetrievalTokenizer,
    make_retrieval_task,
    train_step as retrieval_train_step,
)
from sef_gram.full_system import SEFGRAMConfig, StochasticRecursiveWorldModel


def test_nqueens_solver_and_training_step():
    torch.manual_seed(0)
    solutions = solve_nqueens(4)
    assert len(solutions) == 2
    assert all(solution_is_valid(solution, 4) for solution in solutions)

    device = torch.device("cpu")
    model = StochasticRecursiveWorldModel(
        SEFGRAMConfig(input_dim=4, latent_dim=16, hidden_dim=32, num_actions=4, recursion_depth=4, num_trajectories=4)
    )
    context, actions = batch_from_solutions(solutions, 4, batch_size=4, device=device)
    loss = teacher_forced_policy_loss(model, context, actions)
    assert torch.isfinite(loss)
    loss.backward()

    metrics = evaluate_nqueens(model, n=4, num_trajectories=4, device=device)
    assert set(metrics) == {"valid_solution_rate", "unique_solution_count", "trajectory_diversity", "best_of_k_success"}


def test_terminal_echo_training_step():
    torch.manual_seed(0)
    cfg = TerminalEchoConfig(latent_dim=16, batch_size=2, distractor_len=16, eval_cases=2, device="cpu")
    tokenizer = TinyCharTokenizer()
    model = EchoTerminalModel(tokenizer.vocab_size, cfg.latent_dim)
    samples = [make_terminal_task(cfg.distractor_len)[2] for _ in range(cfg.batch_size)]
    loss = train_step(model, tokenizer, samples, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()


def test_terminal_retrieval_training_step():
    torch.manual_seed(0)
    cfg = RetrievalConfig(latent_dim=16, batch_size=2, distractor_len=16, eval_cases=2, device="cpu")
    tokenizer = RetrievalTokenizer()
    model = RetrievalMemoryModel(tokenizer.vocab_size, cfg.latent_dim)
    tasks = [make_retrieval_task(cfg.distractor_len) for _ in range(cfg.batch_size)]
    loss = retrieval_train_step(model, tokenizer, tasks, torch.device("cpu"), context_lm_weight=0.01)
    assert torch.isfinite(loss)
    loss.backward()


def test_ablation_helpers_smoke():
    torch.manual_seed(0)
    cfg = AblationConfig(device="cpu", terminal_steps=1, terminal_batch_size=2, terminal_distractor_len=16, terminal_eval_cases=2, latent_dim=16)
    tokenizer = RetrievalTokenizer()
    model = NeuralOnlyRetrievalModel(tokenizer.vocab_size, cfg.latent_dim)
    tasks = [make_retrieval_task(cfg.terminal_distractor_len) for _ in range(cfg.terminal_batch_size)]
    loss = neural_retrieval_train_step(model, tokenizer, tasks, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    metrics = evaluate_neural_retrieval(model, tokenizer, cfg, torch.device("cpu"))
    assert "exact_retrieval_accuracy" in metrics
    averaged = mean_metrics([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}])
    assert averaged == {"a": 2.0, "b": 3.0}
