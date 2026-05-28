import torch

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
