from experiments.full_nqueens_benchmark import NQueensRunConfig, run


if __name__ == "__main__":
    run(NQueensRunConfig(board_size=4, train_steps=200, batch_size=32, num_trajectories=16))
