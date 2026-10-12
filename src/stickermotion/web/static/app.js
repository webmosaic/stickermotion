// StickerMotion web UI: drag and drop, progress polling, preview background, confirmations.
(function () {
  "use strict";

  // Upload: drag and drop plus a local preview before sending.
  var drop = document.querySelector("[data-drop]");
  if (drop) {
    var input = drop.querySelector("input[type=file]");
    var preview = drop.querySelector("[data-preview]");
    var title = drop.querySelector("[data-drop-title]");
    var hint = drop.querySelector("[data-drop-hint]");
    var show = function () {
      var file = input.files && input.files[0];
      if (!file) return;
      if (preview.src) URL.revokeObjectURL(preview.src);
      preview.src = URL.createObjectURL(file);
      drop.classList.add("has-file");
      title.textContent = file.name;
      hint.textContent = (file.size / 1024 / 1024).toFixed(1) + " МБ. Натисніть, щоб вибрати інший.";
    };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach(function (name) {
      drop.addEventListener(name, function (event) {
        event.preventDefault();
        drop.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      drop.addEventListener(name, function () { drop.classList.remove("is-over"); });
    });
    drop.addEventListener("drop", function (event) {
      event.preventDefault();
      if (event.dataTransfer && event.dataTransfer.files.length) {
        input.files = event.dataTransfer.files;
        show();
      }
    });
  }

  // Busy buttons: avoid double submits (double payment).
  document.querySelectorAll("form").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      var message = form.getAttribute("data-confirm");
      if (message && !window.confirm(message)) {
        event.preventDefault();
        return;
      }
      var button = event.submitter || form.querySelector("button[type=submit]");
      if (button) {
        window.setTimeout(function () { button.disabled = true; }, 0);
        if (form.hasAttribute("data-upload")) button.textContent = "Завантажуємо…";
      }
    });
  });

  // Variants: the generate button shows the count and the price of the batch.
  var pick = document.querySelector("select[data-variants]");
  if (pick) {
    var update = function () {
      var option = pick.options[pick.selectedIndex];
      var n = Number(pick.value);
      var label = document.querySelector("[data-variants-label]");
      var price = document.querySelector("[data-price]");
      if (label) label.textContent = n > 1 ? " " + n + " варіанти" : "";
      if (price && option) price.textContent = option.getAttribute("data-price");
    };
    pick.addEventListener("change", update);
    update();
  }

  // Preview background switch, remembered per browser.
  var stage = document.querySelector("[data-stage]");
  var buttons = document.querySelectorAll(".switch button[data-bg]");
  var setBg = function (bg) {
    if (!stage) return;
    stage.setAttribute("data-bg", bg);
    buttons.forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-bg") === bg));
    });
    try { localStorage.setItem("sm-bg", bg); } catch (e) { /* storage may be blocked */ }
  };
  buttons.forEach(function (b) {
    b.addEventListener("click", function () { setBg(b.getAttribute("data-bg")); });
  });
  try {
    var saved = localStorage.getItem("sm-bg");
    if (saved) setBg(saved);
  } catch (e) { /* ignore */ }

  // Big preview: WebM with full alpha; browsers that play VP9 without its alpha plane
  // (Safari) or cannot play it at all get the animated WebP made from the same file.
  // Detected by capability: one decoded frame is drawn to a canvas and checked for
  // transparent pixels, no user-agent sniffing.
  var hero = document.querySelector("video[data-hero]");
  if (hero) {
    var useFallback = function () {
      if (hero.getAttribute("data-swapped")) return;
      hero.setAttribute("data-swapped", "1");
      var img = document.createElement("img");
      img.src = hero.getAttribute("data-fallback");
      img.alt = hero.getAttribute("aria-label") || "";
      img.setAttribute("data-hero-fallback", "");
      hero.replaceWith(img);
    };
    var supported = hero.canPlayType('video/webm; codecs="vp9"');
    if (!supported) {
      useFallback();
    } else {
      hero.addEventListener("error", useFallback);
      var probe = function () {
        try {
          var w = 32, h = 32;
          var canvas = document.createElement("canvas");
          canvas.width = w; canvas.height = h;
          var ctx = canvas.getContext("2d", { willReadFrequently: true });
          ctx.drawImage(hero, 0, 0, w, h);
          var data = ctx.getImageData(0, 0, w, h).data;
          for (var i = 3; i < data.length; i += 4) {
            if (data[i] < 250) return; // alpha decoded: keep the WebM
          }
          useFallback();
        } catch (e) { useFallback(); }
      };
      if (hero.readyState >= 2) probe();
      else hero.addEventListener("loadeddata", probe, { once: true });
    }
  }

  // Progress: poll while the sticker is busy, reload the page when the state changes.
  var poll = document.querySelector("[data-poll]");
  var url = poll && poll.getAttribute("data-poll");
  if (url) {
    var state = poll.getAttribute("data-state");
    var tick = function () {
      fetch(url, { headers: { Accept: "application/json" }, credentials: "same-origin" })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
          if (data && data.status !== state) {
            window.location.reload();
            return;
          }
          window.setTimeout(tick, 2000);
        })
        .catch(function () { window.setTimeout(tick, 5000); });
    };
    window.setTimeout(tick, 1500);
  }
})();
