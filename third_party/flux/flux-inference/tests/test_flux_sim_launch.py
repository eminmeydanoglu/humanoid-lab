"""Simulation launch routing and latency budget without a ROS installation."""

import ast
from pathlib import Path


def test_simulation_launch_routes_arm_commands_and_forwards_three_second_budget():
    path = Path(__file__).resolve().parents[1] / "ros2/flux_sim_camera/launch/flux_sim.launch.py"
    tree = ast.parse(path.read_text())
    calls = [item for item in ast.walk(tree) if isinstance(item, ast.Call)]
    declarations = {ast.literal_eval(call.args[0]): ast.literal_eval(next(
        kw.value for kw in call.keywords if kw.arg == "default_value"))
        for call in calls if isinstance(call.func, ast.Name)
        and call.func.id == "DeclareLaunchArgument"
        and ast.literal_eval(call.args[0]) in ("network_timeout_s", "max_chunk_age_s")}
    assert declarations["network_timeout_s"] == "3.0"
    assert declarations["max_chunk_age_s"] == "3.0"
    node = next(call for call in calls if isinstance(call.func, ast.Name)
                and call.func.id == "Node" and any(kw.arg == "name"
                and isinstance(kw.value, ast.Constant) and kw.value.value == "flux_dex3"
                for kw in call.keywords))
    keywords = {kw.arg: kw.value for kw in node.keywords}
    assert ast.literal_eval(keywords["remappings"]) == [("/lowcmd", "/arm_sdk")]
    parameters = keywords["parameters"].elts[0]
    bindings = {ast.literal_eval(key): value.id for key, value in
                zip(parameters.keys, parameters.values)}
    assert bindings["network_timeout_s"] == "network_timeout_s"
    assert bindings["max_chunk_age_s"] == "max_chunk_age_s"


def test_jpeg_preview_uses_raw_topic_without_changing_model_input():
    path = Path(__file__).resolve().parents[1] / "ros2/flux_sim_camera/launch/flux_sim.launch.py"
    tree = ast.parse(path.read_text())
    nodes = {}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Node":
            keywords = {kw.arg: kw.value for kw in call.keywords}
            nodes[ast.literal_eval(keywords["executable"])] = keywords
    jpeg = nodes["camera_jpeg"]
    assert ast.literal_eval(jpeg["package"]) == "flux_sim_camera"
    assert "condition" not in jpeg
    for executable, parameter in (("camera_jpeg", "raw_topic"),
                                  ("camera_bridge", "topic"), ("dex3_node", "camera_topic")):
        params = nodes[executable]["parameters"].elts[0]
        binding = dict(zip((ast.literal_eval(key) for key in params.keys), params.values))
        assert binding[parameter].id == "camera_topic"
