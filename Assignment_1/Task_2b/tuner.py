import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from train2 import train
import ray

num_samples = 10
max_epochs = 400

config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([1])
    }

    
scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2)

result = tune.run(
        tune.with_parameters(train),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

# if ray.util.client.ray.is_connected():
#     # If using Ray Client, we want to make sure checkpoint access
#     # happens on the server. So we wrap `test_best_model` in a Ray task.
#     # We have to make sure it gets executed on the same node that
#     # ``tune.run`` is called on.
#     from ray.util.ml_utils.node import force_on_current_node
#     remote_fn = force_on_current_node(ray.remote(test_best_model))
#     ray.get(remote_fn.remote(best_trial))
# else:
#     test_best_model(best_trial)
