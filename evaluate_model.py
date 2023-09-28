# ignore_header_test
# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from inference import Rollout


def evaluate_model(cfg, logger, model, params, graphs, graph_names):
    rollout = Rollout(logger, cfg, model, params, graphs)
    ep_tot = 0
    eq_tot = 0
    for graph in graph_names:
        rollout.predict(graph)
        rollout.denormalize()
        ep, eq = rollout.compute_errors()
        ep_tot += ep
        eq_tot += eq
    ep_tot = ep_tot / len(graph_names)
    eq_tot = eq_tot / len(graph_names)
    return ep_tot, eq_tot
