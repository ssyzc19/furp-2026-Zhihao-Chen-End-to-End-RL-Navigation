Day12 Evaluation Summary

Checkpoint: ckpt.9

The initial evaluation failed due to a mismatch between the training and evaluation environment configurations.

Investigation of the checkpoint configuration showed that the model had been trained with num_environments=4. Running evaluation with num_environments=1 caused tensor shape mismatch errors inside PointNavResNetPolicy.

A second attempt using num_environments=4 produced an IndexError because the evaluator only created two inference workers during evaluation.

After inspecting the Hydra configuration and inference worker settings, evaluation was rerun with num_environments=2, which successfully completed.

Final Results:

* Success Rate (SR): 0.8000
* SPL: 0.6465
* Distance To Goal: 0.1160
* Average Reward: 4.5619

Ten evaluation episodes were executed, producing eight successful navigation trajectories and two failure cases. Evaluation videos were exported for qualitative analysis.
