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

import os
import numpy as np
import dgl
from tqdm import tqdm
import json
import shutil
import copy
import vtk_tools as vtkt
import graph_tools as grpt
import scipy
import torch as th
from dgl import NodeShuffle

dgl.seed(11)

def add_field(graph, field, field_name, offset=0):
    """
    Add time-dependent fields to a DGL graph.

    Add time-dependent scalar fields as graph node features. The time-dependent
    fields are stored as n x 1 x m Pytorch tensors, where n is the number of
    graph nodes and m the number of timesteps.

    Arguments:
        graph: DGL graph
        field: dictionary with (key: timestep, value: field value)
        field_name (string): name of the field
        offset (int): number of timesteps to skip.
                      Default: 0 -> keep all timesteps
    """
    timesteps = [float(t) for t in field]
    timesteps.sort()
    dt = timesteps[1] - timesteps[0]
    T = timesteps[-1]

    # we use the third dimension for time
    field_t = th.zeros((list(field.values())[0].shape[0], 1, len(timesteps) - offset))

    times = [t for t in field]
    times.sort()
    times = times[offset:]

    for i, t in enumerate(times):
        f = th.tensor(field[t], dtype=th.float32)
        field_t[:, 0, i] = f

    graph.ndata[field_name] = field_t
    graph.ndata["dt"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * dt, (-1, 1, 1)
    )
    graph.ndata["T"] = th.reshape(
        th.ones(graph.num_nodes(), dtype=th.float32) * T, (-1, 1, 1)
    )


def load_vtp(file, input_dir):
    """
    Load vtp file.

    Arguments:
        file (string): file name
        input_dir (string): path to input_dir

    Returns:
        dictionary containing point data (key: name, value: data)
        n x 3 numpy array of point coordinates
        numpy array containing indices of source nodes for every edge
        numpy array containing indices of dest nodes for every edge

    """
    soln = vtkt.read_geo(input_dir + "/" + file)
    point_data, _, points = vtkt.get_all_arrays(soln.GetOutput())
    edges1, edges2 = vtkt.get_edges(soln.GetOutput())

    # lets check for nans and delete points if they appear
    ni = np.argwhere(np.isnan(point_data["area"]))
    if ni.size > 0:
        for i in ni[0]:
            indices = np.where(edges1 >= i)[0]
            edges1[indices] = edges1[indices] - 1

            indices = np.where(edges2 >= i)[0]
            edges2[indices] = edges2[indices] - 1

            indices = np.where(edges1 == edges2)[0]
            edges1 = np.delete(edges1, indices)
            edges2 = np.delete(edges2, indices)

            points = np.delete(points, i, axis=0)
            for ndata in point_data:
                point_data[ndata] = np.delete(point_data[ndata], i)

    return point_data, points, edges1, edges2


def resample_time(field, timesteps, period, shift=0):
    """
    Resample timesteps.

    Given a time-dependent field distributed over graph nodes, this function
    resamples the field in time using B-spline interpolation at every node.

    Arguments:
        field: dictionary containing the field for all timesteps
               (key: timestep, value: n-dimensional numpy array)
        timesteps (int): number of timesteps. Default -> 100
        period (float): period of the simulation (one cardiac cycle in seconds).
        shift (float): apply shift (s) to start at the beginning of the systole.
                       Default -> 0

    Returns:
        dictionary containing the field for all resampled timesteps
            (key: timestep, value: n-dimensional numpy array)
    """
    original_timesteps = [t for t in field]
    original_timesteps.sort()

    t0 = original_timesteps[0]
    T = original_timesteps[-1]

    t = [t0 + shift]
    nnodes = field[t0].size
    # allocating space for the initial condition. This is a dictionary where the key
    # is the timestep and the value is the vector of nodal values.
    resampled_field = {} # {t0 + shift: np.zeros(nnodes)}

    t = np.linspace(t0, t0 + period, timesteps)
    for t_ in t:
        resampled_field[t_] = np.zeros(nnodes)

    #print(len(t))

    for inode in range(nnodes):
        values = []
        for time in original_timesteps:
            values.append(field[time][inode])

        tck, _ = scipy.interpolate.splprep([values], u=original_timesteps, s=0)
        values_interpolated = scipy.interpolate.splev(t, tck)[0]

        for i, time in enumerate(t):
            resampled_field[time][inode] = values_interpolated[i]

    return resampled_field


def generate_datastructures(vtp_data, dataset_info, resample_perc):
    """
    Generate data structures for graph generation from vtp data.

    Arguments:
        vtp_data: tuple containing data extracted from the vtp using load_vtp
        dataset_info: dictionary containing dataset information
        resample_perc: percentage of points in the original vtp file we keep
                       (between 0 and 1)
    Returns:
        dictionary containing graph data (key: field name, value: data)
    """
    point_data, points, edges1, edges2 = vtp_data
    point_data["tangent"] = grpt.generate_tangents(points, point_data["BranchIdTmp"])
    # first node is the inlet by convention
    inlet = [0]
    outlets = grpt.find_outlets(edges1, edges2)

    indices = {"inlet": inlet, "outlets": outlets}

    success = False

    while not success:
        try:
            sampled_indices, points, edges1, edges2, _ = grpt.resample_points(
                points.copy(),
                edges1.copy(),
                edges2.copy(),
                indices,
                resample_perc,
                remove_caps=3,
            )
            success = True
        except Exception as e:
            print(e)
            resample_perc = np.min([resample_perc * 2, 1])

    for ndata in point_data:
        point_data[ndata] = point_data[ndata][sampled_indices]

    inlet = [0]
    outlets = grpt.find_outlets(edges1, edges2)

    indices = {"inlet": inlet, "outlets": outlets}

    pressure = vtkt.gather_array(point_data, "pressure")
    flowrate = vtkt.gather_array(point_data, "flow")
    if len(flowrate) == 0:
        flowrate = vtkt.gather_array(point_data, "velocity")

    times = [t for t in pressure]
    timestep = float(dataset_info[file.replace(".vtp", "")]["dt"])
    period = float(dataset_info[file.replace(".vtp", "")]["T"])
    for t in times:
        pressure[t * timestep] = pressure[t]
        flowrate[t * timestep] = flowrate[t]
        del pressure[t]
        del flowrate[t]

    # scale pressure to be mmHg
    for t in pressure:
        pressure[t] = pressure[t] / 1333.2

    times = [t for t in pressure]

    sampling_indices = np.arange(points.shape[0])
    graph_data = {
        "point_data": point_data,
        "points": points,
        "edges1": edges1,
        "edges2": edges2,
        "sampling_indices": sampling_indices,
        "pressure": pressure,
        "flowrate": flowrate,
        "timestep": timestep,
        "times": times,
        "period": period,
    }

    return graph_data


def add_time_dependent_fields(
    graph, graph_data, do_resample_time=False, timesteps=101, ncopies=1
):
    """
    Add time-dependent data to a graph containing static data. This function
    can be used to create multiple graphs from a single trajectory by
    specifying do_resample_time and providing a number of copies > 1. In this
    case, every graph trajectories starts at a different offset from the
    starting time.

    Arguments:
        graph: a DGL graph.
        graph_data: dictionary containing graph_data (created using
                    generate_datastructures)
        do_resample_time (bool): specify whether we should resample the
                                 the timesteps. Default -> False
        timesteps (int): number of timesteps to use for resampling. Default -> 101
        ncopies: number of copies to generate from a single trajectory (for
                data augmentation). Default -> 1

    Returns:
        list of 'copies' graphs.
    """

    graphs = []
    for icopy in range(ncopies):
        c_pressure = {}
        c_flowrate = {}

        si = graph_data["sampling_indices"]
        for t in graph_data["times"]:
            c_pressure[t] = graph_data["pressure"][t][si]
            c_flowrate[t] = graph_data["flowrate"][t][si]

        if do_resample_time:
            shift = dataset_info[fname]["time_shift"]
            # some simulations in the dataset need to be translated in the time because
            # the start of the simuluation does not coincide with the onset of the
            # systole. In the json associated with the dataset, we added a parameter
            # to determine how much the translation should be. Here we are also adding
            # a small shift to create additional trajectories for data augmentation.
            dt = graph_data["period"] / (timesteps - 1)
            data_augmentation_shift = dt / ncopies * icopy
            actual_shift = data_augmentation_shift + shift
            c_pressure = resample_time(
                c_pressure, timesteps, period=graph_data["period"], shift=actual_shift
            )
            c_flowrate = resample_time(
                c_flowrate, timesteps, period=graph_data["period"], shift=actual_shift
            )

        new_graph = copy.deepcopy(graph)
        add_field(new_graph, c_pressure, "pressure")
        #print(new_graph.ndata["pressure"].shape)
        add_field(new_graph, c_flowrate, "flowrate")
        graphs.append(new_graph)

        # Create a shuffled version of the graph
        #shuffled_graph = NodeShuffle()(new_graph)
        #graphs.append(shuffled_graph)

    return graphs


"""
The main function reads all vtps files from the folder specified in input_dir
and generates DGL graphs. The graphs are saved in output_dir.
"""
if __name__ == "__main__":
    input_dir = "raw_dataset/vtps"
    output_dir = "raw_dataset/graphs/"

    dataset_info = json.load(open(input_dir + "/dataset_info.json"))

    files = os.listdir(input_dir)

    print("Processing all files in {}".format(input_dir))
    print("File list:")
    print(files)
    for file in tqdm(files, desc="Generating graphs", colour="green"):
        if ".vtp" in file and "s" in file:
            vtp_data = load_vtp(file, input_dir)
            fname = file.replace(".vtp", "")

            graph_data = generate_datastructures(
                vtp_data, dataset_info, resample_perc=0.06
            )
            static_graph = grpt.generate_graph(
                graph_data["point_data"],
                graph_data["points"],
                graph_data["edges1"],
                graph_data["edges2"],
                add_boundary_edges=True,
                rcr_values=dataset_info[fname],
                debug=False,
                pivotalnodes=True
            )

            # Create shuffled and original graphs
            graphs = add_time_dependent_fields(
                static_graph, graph_data, do_resample_time=True, timesteps=41, ncopies=4
            )

            for i, graph in enumerate(graphs):
                filename = file.replace(".vtp", "." + str(i) + ".grph")
                dgl.save_graphs(output_dir + filename, graph)

    shutil.copy(input_dir + "/dataset_info.json", output_dir + "/dataset_info.json")