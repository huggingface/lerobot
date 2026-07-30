const shell = document.querySelector(".app-shell");
const $ = (selector) => document.querySelector(selector);
let attachedImage = "";
let toastTimer;
let plannerTimer;
let plannerBusy = false;
let demoMode = false;

function toast(text) {
  const node = $("#toast");
  node.textContent = text;
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 2600);
}

async function api(path, payload) {
  if (demoMode) {
    await new Promise((resolve) => setTimeout(resolve, 350));
    if (payload.kind === "vqa") {
      return { answer: payload.image_url
        ? "I can see the attached reference image. Connect a VQA-capable checkpoint to ground this answer in model inference."
        : "This Space is showing a recorded rollout preview. Start a live language runtime to ground answers in the current camera frame." };
    }
    if (payload.kind === "planner") return { answer: "Move the gripper toward the target handle." };
    return { ok: true };
  }
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function activateDemoMode() {
  if (demoMode) return;
  demoMode = true;
  const stream = $("#stream");
  const video = document.createElement("video");
  video.id = "demo-stream";
  video.src = "/playground-demo.mp4";
  video.autoplay = true;
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  stream.replaceWith(video);
  $("#connection").textContent = "Preview · replay";
  $(".live-pill").classList.remove("connected");
  $("#active-task").textContent = $("#prompt").value;
  $("#subtask").textContent = "Recorded policy rollout";
  $("#memory").textContent = "Connect the runtime for live state";
  $("#policy").selectedOptions[0].textContent = "Preview checkpoint";
}

function setPanel(name, open) {
  if (name === "chat" && open) {
    shell.dataset.blogOpen = "false";
    $("#blog-toggle").setAttribute("aria-expanded", "false");
  }
  if (name === "blog" && open) {
    shell.dataset.chatOpen = "false";
    $("#chat-toggle").setAttribute("aria-expanded", "false");
  }
  shell.dataset[`${name}Open`] = String(open);
  $(`#${name}-toggle`).setAttribute("aria-expanded", String(open));
}

$("#chat-toggle").onclick = () => setPanel("chat", shell.dataset.chatOpen !== "true");
$("#chat-close").onclick = () => setPanel("chat", false);
$("#blog-toggle").onclick = () => setPanel("blog", shell.dataset.blogOpen !== "true");
$("#blog-close").onclick = () => setPanel("blog", false);

$("#unfreeze").onclick = () => {
  const input = $("#prompt");
  const willUnlock = input.hasAttribute("readonly");
  input.toggleAttribute("readonly", !willUnlock);
  $("#unfreeze").textContent = willUnlock ? "Lock" : "Unfreeze";
  if (willUnlock) input.focus();
};

$("#run").onclick = async () => {
  const task = $("#prompt").value.trim();
  if (!task) return toast("Enter a task first");
  try {
    await api("/api/command", { kind: "action", text: task, scene: $("#task").value });
    toast("Policy running");
  } catch (error) { toast(error.message); }
};
$("#pause").onclick = () => api("/api/command", { kind: "pause" }).then(() => toast("Paused")).catch((e) => toast(e.message));
$("#reset").onclick = () => api("/api/command", { kind: "reset" }).then(() => toast("Scene reset")).catch((e) => toast(e.message));
$("#task").onchange = (event) => {
  const readable = event.target.selectedOptions[0].text.replace(/([a-z])([A-Z])/g, "$1 $2");
  $("#prompt").value = readable;
};
$("#benchmark").onchange = (event) => {
  if (event.target.value !== "robocasa") {
    toast(`${event.target.selectedOptions[0].textContent} is catalogued but not enabled by this runtime`);
    event.target.value = "robocasa";
  }
};
$("#policy").onchange = (event) => {
  if (event.target.value !== "runtime") {
    toast("Checkpoint hot-swap requires starting a new runtime session");
    event.target.value = "runtime";
  }
};

function addMessage(role, text, imageUrl = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("span");
  avatar.textContent = role === "user" ? "YOU" : "LR";
  const body = document.createElement("div");
  const name = document.createElement("b");
  name.textContent = role === "user" ? "You" : "LeRobot";
  const copy = document.createElement("p");
  copy.textContent = text;
  body.append(name, copy);
  if (imageUrl) {
    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = "VQA attachment";
    body.append(image);
  }
  article.append(avatar, body);
  $("#messages").append(article);
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

$("#image-button").onclick = () => {
  $("#image-row").hidden = !$("#image-row").hidden;
  if (!$("#image-row").hidden) $("#image-url").focus();
};
$("#attach-url").onclick = () => {
  const value = $("#image-url").value.trim();
  try { new URL(value); } catch { return toast("Enter a valid image URL"); }
  attachedImage = value;
  $("#image-preview img").src = value;
  $("#image-preview span").textContent = value;
  $("#image-preview").hidden = false;
  $("#image-row").hidden = true;
};
$("#image-preview button").onclick = () => {
  attachedImage = "";
  $("#image-preview").hidden = true;
  $("#image-url").value = "";
};

$("#chat-form").onsubmit = async (event) => {
  event.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  const kind = $("#chat-mode").value;
  const imageUrl = attachedImage;
  addMessage("user", text, imageUrl);
  input.value = "";
  $("#image-preview button").click();
  try {
    if (kind === "action") {
      await api("/api/command", { kind: "action", text });
      addMessage("assistant", `Running: ${text}`);
    } else {
      const response = await api("/api/chat", { kind: "vqa", text, image_url: imageUrl || undefined });
      addMessage("assistant", response.answer || "The policy returned no answer.");
    }
  } catch (error) {
    addMessage("assistant", `I couldn't complete that request: ${error.message}`);
  }
};

function configurePlanner() {
  clearInterval(plannerTimer);
  plannerTimer = undefined;
  if (!$("#planner-enabled").checked) return;
  const intervalMs = Math.max(250, Number($("#planner-rate").value || 1) * 1000);
  const plan = async () => {
    if (plannerBusy) return;
    plannerBusy = true;
    try {
      const response = await api("/api/chat", {
        kind: "planner",
        text: $("#prompt").value.trim(),
        planner_prompt: $("#planner-prompt").value.trim(),
      });
      if (response.answer) {
        $("#subtask").textContent = response.answer;
        addMessage("assistant", `Next subtask: ${response.answer}`);
      }
    } catch (error) {
      $("#planner-enabled").checked = false;
      clearInterval(plannerTimer);
      toast(`Planner stopped: ${error.message}`);
    } finally {
      plannerBusy = false;
    }
  };
  plan();
  plannerTimer = setInterval(plan, intervalMs);
}
$("#planner-enabled").onchange = configurePlanner;
$("#planner-rate").onchange = configurePlanner;

async function refresh() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    const data = await response.json();
    const state = data.state || {};
    $(".live-pill").classList.toggle("connected", Boolean(data.connected));
    $("#connection").textContent = data.connected ? (state.mode === "action" ? "Live · running" : "Live · paused") : "Waiting for runtime";
    $("#active-task").textContent = state.task || "No task selected";
    $("#subtask").textContent = state.language_context?.subtask || "Waiting for policy";
    $("#memory").textContent = state.language_context?.memory || "No observations yet";
    $("#policy").selectedOptions[0].textContent = data.policy_path || "Connected checkpoint";
    if (data.blog_url && !$("#blog-frame").src) $("#blog-frame").src = data.blog_url;
  } catch {
    activateDemoMode();
  }
}
refresh();
setInterval(refresh, 900);
