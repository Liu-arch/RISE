import json,re,math
SYSTEM = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element.
Don't click boxes on their edges unless asked.
* don't use any other computer use tool like type, key, scroll, left_click_drag and so on.
* you can only use the left_click and mouse_move action to interact with the computer. if you can't find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform.
The available actions are:
* ~mouse_move~: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* ~left_click~: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.
Required only by ~action=mouse_move~ and ~action=left_click~.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""".replace("~", chr(96))
INFEASIBLE = """Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return <tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}
</tool_call>"""
def prompt(benchmark, arm, observable, definitions):
    assert arm == "A" and benchmark in ("osworldg", "screenspot")
    return SYSTEM + ("\n" + INFEASIBLE if benchmark == "osworldg" else ""), observable["instruction"]

def native_action(raw):
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S)
    if not blocks:
        return {}, "missing_tool_call"
    try:
        call = json.loads(blocks[-1])
    except (ValueError, TypeError):
        return {}, "invalid_tool_call_json"
    if not isinstance(call, dict) or call.get("name") != "computer_use" or not isinstance(call.get("arguments"), dict):
        return {}, "wrong_tool_name_or_arguments"
    args = call["arguments"]
    if args.get("action") == "terminate" and args.get("status") == "failure":
        return {"type": "wait", "status": "failure"}, None
    if args.get("action") not in ("left_click", "mouse_move"):
        return {}, "unsupported_action"
    xy = args.get("coordinate")
    if not isinstance(xy, list) or len(xy) != 2 or any(type(z) not in (int, float) or not math.isfinite(z) or not 0 <= z <= 1000 for z in xy):
        return {}, "invalid_coordinate"
    return {"type": "click", "x_norm": xy[0]/1000, "y_norm": xy[1]/1000}, None
