# ----------------------------------- Introduction -----------------------------------
introduction = "Imagine you are an autonomous robot in an indoor environment for instruction following task. You should follow the instruction and then predict a navigable goal pixel ratio in the image. \n"

introduction_pixel_in_text = (
    "You are an indoor navigation agent following a language instruction. "
    "Given the current egocentric image, predict the next navigation target as a 2D pixel coordinate in the image. "
    "Output exactly one coordinate pair (X, Y), where X and Y are integers in the range [0, 999]. "
    "The coordinate represents the normalized horizontal and vertical position in the image. "
    "Use the following exact coordinates for special actions: "
    "TURN_LEFT = (000, 500), TURN_RIGHT = (999, 500), STOP = (500, 999). "
    "For forward navigation, output the pixel that best indicates the next navigable direction or goal in the current view. "
    "Prefer predicting a farthest navigablepixel on the visible image region whenever a navigable target is visible. "
    "If no clear forward navigable target is visible, output a pixel on the left padding or right padding to indicate turning. "
    "If the agent is within 1.5 meters of the destination, output the STOP coordinate on the bottom padding. "
    "Return only the coordinate pair in the format XXX, YYY."
)
introduction_action =  "You are an autonomous indoor navigation robot. Given an instruction and the current observation, choose the single best next action to follow the instruction.\n Action space (choose exactly one): MOVE_FORWARD (25 cm), TURN_LEFT (30°), TURN_RIGHT (30°), STOP. Termination rule: if the robot is within 1.5 m of the final destination, choose STOP.\n Return only the action token (MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP)."
introduction_4action =  "You are an autonomous indoor navigation robot. Given an instruction and the current observation, choose next four best next actions to follow the instruction.\n Action space: MOVE FORWARD (25 cm), TURN LEFT (15°), TURN RIGHT (15°), STOP. Termination rule: if the robot is within 1.5 m of the final destination, choose STOP.\n Return only the action token (MOVE FORWARD / TURN LEFT / TURN RIGHT / STOP)."
introduction_action_pixel = (
    "Imagine you are an autonomous robot performing an indoor instruction-following task. "
    "Given the navigation instruction and the current image, predict the next step.\n\n"
    "The next step consists of:\n"
    "- an action from {FORWARD, LEFT, RIGHT, STOP}\n"
    "- a goal pixel coordinate (X, Y) when and only when the action is FORWARD\n\n"
    "Constraints:\n"
    "- X and Y must be integers in the range [0, 999]\n"
    "- The pixel (X, Y) must indicate a navigable target in the current image\n"
    "- If the action is LEFT, RIGHT, or STOP, output only the action\n"
    "- X and Y must each be formatted as a three-digit integer string with leading zeros if necessary (e.g., 1 -> 001, 12 -> 012, 123 -> 123)\n"
    "Output format:\n"
    "- FORWARD (X, Y)\n"
    "- LEFT\n"
    "- RIGHT\n"
    "- STOP\n"
)

# ----------------------------------- RGB image -----------------------------------
RGB = "- Current Step egocentric RGB image <image> \n"
RGB_PADDED  = "- Current Step egocentric RGB image <image> , where the left/right grey padding indicates left or right turns.\n"
RGB_PADDED_DOWN = "- Current Step egocentric RGB image <image> , where the left/right gray padding indicates left/right turns, and the bottom padding indicates stop. \n"
RGB_PADDED_DOWN_LAST = "- Current Step egocentric RGB image, where the left/right gray padding indicates left/right turns, and the bottom padding indicates stop <image>. \n"
# ---------------- ------------------- Occupancy Map -----------------------------------
# OCCUPANCY_EXPLORED_MAP = "- BEV grid map <image> showing free (white), occupied (black), unexplored (gray), robot pose/heading (red arrow), and past trajectory (blue line).\n"
OCCUPANCY_EXPLORED_MAP_SINGLE_COLOR = "- BEV grid map <image> showing free (white), occupied (black), unexplored (gray), robot pose/heading (red arrow), and past trajectory (blue line).\n"
OCCUPANCY_EXPLORED_MAP_MULTIPLE_COLOR = "- BEV grid map <image> showing free (white), occupied (black), unexplored (gray), robot pose/heading (red arrow), and past trajectory colored by recency: red (latest), then blue/green/yellow/magenta.\n"



# ------------------------------------ History: interval ------------------------------------
def build_interval_history_prompt(number_image: int, interval: int, max_image_number: int) -> str:
	"""Build interval-history prompt text from dynamic arguments.

	Example:
		number_image=2, interval=1, max_image_number=5
		-> ... <image>(t-1), <image>(t-2)
	"""
	interval = max(1, int(interval))
	max_image_number = max(0, int(max_image_number))
	number_image = max(0, min(int(number_image), max_image_number))

	if number_image == 0:
		return "- History Images: No history images are available for this step.\n"

	timestep_word = "timestep" if interval == 1 else "timesteps"
	sequence = ", ".join(
		f"<image>(t-{k * interval})" for k in range(1, number_image + 1)
	)
	return (
		f"- History Images: Previous {number_image} egocentric RGB images sampled every "
		f"{interval} {timestep_word} backward from the current step: {sequence}\n"
	)


# Backward-compatible alias for callers that use the old spelling.
def build_interval_history_prompt_compat(number_image: int, internval: int, max_image_number: int) -> str:
	return build_interval_history_prompt(number_image, internval, max_image_number)


# HIS_1INTERVAL = build_interval_history_prompt(number_image=5, interval=1, max_image_number=5)
# HIS_2INTERVAL = build_interval_history_prompt(number_image=5, interval=2, max_image_number=5)
# HIS_3INTERVAL = build_interval_history_prompt(number_image=5, interval=3, max_image_number=5)
# HIS_4INTERVAL = build_interval_history_prompt(number_image=5, interval=4, max_image_number=5)
# HIS_5INTERVAL5 = build_interval_history_prompt(number_image=5, interval=5, max_image_number=5)
# HIS_5INTERVAL1 = build_interval_history_prompt(number_image=1, interval=5, max_image_number=5)
# HIS_5INTERVAL3 = build_interval_history_prompt(number_image=3, interval=5, max_image_number=5)
# HIS_1INTERVAL8 = build_interval_history_prompt(number_image=8, interval=1, max_image_number=8)


# HIS by sampling
def build_uniform_history_prompt(number_image: int, max_image_number: int = 8) -> str:
	"""Build uniform-sampling history prompt with matching <image> token count."""
	max_image_number = max(0, int(max_image_number))
	number_image = max(0, min(int(number_image), max_image_number))

	if number_image == 0:
		return "- History Images: No history images are available for this step.\n"

	tokens = []
	last_index = number_image - 1
	for idx in range(number_image):
		if idx == 0:
			suffix = "(latest)"
		elif idx == last_index:
			suffix = "(oldest)"
		else:
			suffix = ""
		tokens.append(f"<image>{suffix}")

	return (
		f"- History Images: Previous {number_image} egocentric RGB images uniformly sampled over the past "
		f"trajectory from the current step back to episode start: {', '.join(tokens)}\n"
	)


# Backward-compatible constant for callers that still expect a fixed string.
HIS_UNIFORM_SAMPLING = build_uniform_history_prompt(number_image=8, max_image_number=8)


def build_diff_frequency_history_prompt(
	number_image: int,
	max_image_number: int = 8,
	recent_focus_number: int = 4,
) -> str:
	"""Build mixed-frequency history prompt with matching <image> token count."""
	max_image_number = max(0, int(max_image_number))
	number_image = max(0, min(int(number_image), max_image_number))
	recent_focus_number = max(0, int(recent_focus_number))

	if number_image == 0:
		return "- History Images: No history images are available for this step.\n"

	tokens = []
	last_index = number_image - 1
	for idx in range(number_image):
		if idx == 0:
			suffix = "(latest)"
		elif idx == last_index:
			suffix = "(oldest)"
		else:
			suffix = ""
		tokens.append(f"<image>{suffix}")

	recent_count = min(recent_focus_number, number_image)
	older_count = number_image - recent_count
	if older_count > 0:
		density_text = (
			f"dense for the {recent_count} most recent and sparse for the {older_count} older frames"
		)
	else:
		density_text = "dense for all available recent frames"

	return (
		f"- History Images: Previous {number_image} egocentric RGB frames sampled backward with a mixed "
		f"frequency ({density_text}): {', '.join(tokens)}\n"
	)


# Backward-compatible constant for callers that still expect a fixed string.
HIS_DIFF_FRE_SAMPLING = build_diff_frequency_history_prompt(
	number_image=8,
	max_image_number=8,
	recent_focus_number=4,
)
# HIS by keyframe
HIS_KEYFRAME_SINGLE_COLORED_ = "- History Images: Previous 5 egocentric RGB keyframes from the past trajectory, with blue trajectory showing movement trajectory: <image>(latest), <image>, <image>, <image>, <image>(oldest)\n"

def _format_history_sequence(num_images: int) -> str:
	"""Return comma-separated <image> tokens with latest/oldest tags."""
	num_images = int(max(0, num_images))
	if num_images == 0:
		return ""

	tokens = []
	last_index = num_images - 1
	for idx in range(num_images):
		if idx == 0:
			suffix = "(latest)"
		elif idx == last_index:
			suffix = "(oldest)"
		else:
			suffix = ""
		tokens.append(f"<image>{suffix}")
	return ", ".join(tokens)

def _build_keyframe_prompt(prefix: str, num_images: int) -> str:
	"""Compose keyframe prompt text with the requested image count."""
	num_images = int(max(0, num_images))
	if num_images == 0:
		return "- History Images: No keyframe history images are available for this step.\n"

	sequence = _format_history_sequence(num_images)
	return f"{prefix.format(count=num_images)} {sequence}\n"


def HIS_KEYFRAME_NO_COLORED(num_images: int) -> str:
	return _build_keyframe_prompt(
		"- History Images: Previous {count} egocentric RGB keyframes from the past trajectory:",
		num_images,
	)


def HIS_KEYFRAME_SINGLE_COLORED(num_images: int) -> str:
	return _build_keyframe_prompt(
		"- History Images: Previous {count} egocentric RGB keyframes from the past trajectory, with blue trajectory showing movement trajectory:",
		num_images,
	)


def HIS_KEYFRAME_MULTIPLE_COLORED(num_images: int) -> str:
	return _build_keyframe_prompt(
		"- History Images: Previous {count} egocentric RGB keyframes from the past trajectory, with the overlaid trajectory colored by recency: red (latest), then blue/green/yellow, and magenta (oldest):",
		num_images,
	)

# History Action
HIS_FULL_ACTION = "- History Action: A text description summarizing all past actions taken in the episode from oldest to newest: "
HIS_FULL_ACTION_5 = "- Recent 5 History Actions: A text description summarizing recent 5 past actions taken in the episode from oldest to newest: "




# Output Format
OUTPUT_ACTION = "Output: only one action from {MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP}."
OUTPUT_ACTION_4 = "Output: the next four actions, each from {MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP}, separated by commas. Example: MOVE_FORWARD, TURN_LEFT, MOVE_FORWARD, STOP"
OUTPUT_JSON = "Output: a pixel coordinates (X, Y) in json format - Example: {'point_2d': [ddd, ddd]}"
OUTPUT_PIXEL = "Output: a pixel coordinates (X, Y) - Example: 123, 456"
OUTPUT_ACTION_PIXEL = "Output: an action (FORWARD, LEFT, RIGHT, STOP) and a pixel coordinates (X, Y) - Example: FORWARD (DDD, DDD)/LEFT/RIGHT/STOP"
HIS_ACTION_5 = "- History Action: A text description summarizing the most recent 5 past actions taken in the episode from oldest to newest: "


# Output
OUTPUT_ACTION = "Output: only one action from {MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP}."
OUTPUT_JSON = "Output: a pixel coordinates (X, Y) in json format - Example: {'point_2d': [ddd, ddd]}"
OUTPUT_PIXEL = "Output: a pixel coordinates (X, Y) in json format - Example: 123, 456"
OUTPUT_ACTION_PIXEL = "Output: an action (FORWARD, LEFT, RIGHT, STOP) and a pixel coordinates (X, Y) - Example: FORWARD (DDD, DDD)/LEFT/RIGHT/STOP"
OUTPUT_4ACTION = "Output: four next actions from {MOVE FORWARD, TURN LEFT, TURN RIGHT, STOP}."

# ----------------------------------- Note -----------------------------------
# relationship between occupancy and history image
NOTE = f"\nNote: \n"
NOTE_OCCU_HISTORY_SINGLE_COLOR = "- BEV occupancy map summarized trajectory accumulated from history images.\n"
NOTE_OCCU_HISTORY_MULTIPLE_COLOR = "- Relationship: BEV occupancy map summarized trajectory accumulated from history images with corresponding colors.\n"

DESCRIPTION = "This is what you have done: "


