# Prompting

The action description is the main thing you control. The automatic description is usually a good start; these tips help when you edit it or write your own (`--prompt` in the CLI, the text field in the web UI).

Fixed requirements (static camera, flat background, return to the starting pose, same style, same skin and hand colours, mouth and hands as in the drawing) are appended in code. You only describe the action.

## Write only positive statements

Wan 2.2 fast has **no negative prompt** and does not understand negation. Words like "not", "no" or "without" are ignored, and the action they name becomes a cue. "She does not laugh" makes laughing more likely.

Describe the state you want instead:

| Instead of | Write |
|---|---|
| she does not laugh | her lips stay closed, the mouth small as in the drawing |
| no big mouth movements | the mouth keeps its size and shape from the drawing |
| the hand does not move | the raised hand stays still in the same gesture |
| no camera movement | (already appended in code) |
| he does not open his eyes | his eyes stay closed and relaxed |

## Avoid open-mouth actions unless you need them

Actions such as yawning, singing, shouting or laughing make the model open the mouth wide, often much wider than the character's style allows. If the action does not need it, do not name it at all. "Sleepy, slowly nods and closes the eyes" works better than "yawns" for a calm character.

## Keep gesturing hands still, animate the face and head

Hands with a gesture (a thumbs-up, a peace sign, a raised fist, a heart) are where video models fail most: fingers merge, change count or colour. Keep such a hand still and put the motion into the face and head:

- "The raised thumbs-up hand stays still. She blinks, smiles a little wider and tilts her head to the side."
- "The fist stays in place. His eyebrows lift and he nods once."

A simple open hand that waves is usually fine.

## Small amplitude

Stickers loop every 2-3 seconds and emotes are viewed at 28 px. Small, clear motions read better and loop more cleanly than big ones:

- one nod, one blink, a slight head tilt, a short wave;
- "slightly", "gently", "a little", "slowly";
- one main action, at most two.

Large motions (jumping, turning around, big arm swings) push parts of the character out of the frame and make the loop seam visible.

## Return to the start

The model gets the same image as first and last frame. Actions that naturally return (nod, blink, wave, breathe, tilt and back) loop best. Actions that end somewhere else (stands up, walks away, drops something) fight the loop.

## Decor

With "animate decor separately" on (default), the decor is not in the video at all, so do not describe it; its motion is procedural. With it off, you can describe the decor ("the stars twinkle softly"), but it will move together with the scene.

## Generate variants and pick the best

The same prompt can give a calm motion with one seed and a wide "singing" mouth with another. Generate 2-3 variants (about $0.05 each at 480p) and choose. A liked clip is stored; you can re-export it for free later, or repeat it with its seed in the CLI (`--seed`).

## Examples

- "The cat slowly blinks twice and gently tilts its head to the right, its mouth stays small and closed as in the drawing."
- "The girl waves her open right hand twice at shoulder height and smiles softly with closed lips."
- "The raised peace-sign hand stays still. The boy winks with his left eye and nods once."
- "The sleepy character slowly lowers its head, its eyes close, and it rises back to the starting pose."
