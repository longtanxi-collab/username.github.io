const $ = (id) => document.getElementById(id);

const VISION_TASKS_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const state = {
  landmarker: null,
  modelReady: false,
  fallbackReady: true,
  metrics: null,
  image: null,
  latestReport: null
};

const refs = {
  modelStatus: $("modelStatus"),
  scanMessage: $("scanMessage"),
  dropZone: $("dropZone"),
  photoInput: $("photoInput"),
  photoPreview: $("photoPreview"),
  canvas: $("overlayCanvas")
};

function setStatus(text, type = "") {
  refs.modelStatus.textContent = text;
  refs.modelStatus.className = `status-pill ${type}`.trim();
}

function setMessage(text, type = "") {
  refs.scanMessage.textContent = text;
  refs.scanMessage.className = `tech-note ${type}`.trim();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function barPercent(value, min, max) {
  return `${clamp(Math.round(((value - min) / (max - min)) * 100), 7, 100)}%`;
}

async function initFaceLandmarker() {
  try {
    setStatus("模型加载中");
    const { FaceLandmarker, FilesetResolver } = await import(VISION_TASKS_URL);
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    const createLandmarker = (delegate) => FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate
      },
      runningMode: "IMAGE",
      numFaces: 5
    });

    try {
      state.landmarker = await createLandmarker("GPU");
    } catch (gpuError) {
      console.warn("GPU delegate unavailable, falling back to CPU.", gpuError);
      state.landmarker = await createLandmarker("CPU");
    }

    state.modelReady = true;
    setStatus("AI已就绪", "ready");
    setMessage("模型已就绪。上传正脸照片后，将在本地完成478个面部关键点识别。", "success");
  } catch (error) {
    state.modelReady = false;
    setStatus("本地估算可用", "ready");
    setMessage("精准关键点模型暂不可用，已启用本地结构估算模式。照片仍只在本地浏览器处理，不上传服务器。", "warning");
    console.error(error);
  }
}

function readImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = reject;
    reader.onload = () => {
      const image = new Image();
      image.onload = () => resolve({ image, dataUrl: reader.result });
      image.onerror = reject;
      image.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

function chooseLargestFace(faces) {
  let maxArea = -1;
  let index = 0;

  faces.forEach((face, faceIndex) => {
    const xs = face.map((point) => point.x);
    const ys = face.map((point) => point.y);
    const area = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
    if (area > maxArea) {
      maxArea = area;
      index = faceIndex;
    }
  });

  return index;
}

function calculateMetrics(landmarks) {
  const faceTop = landmarks[10];
  const chin = landmarks[152];
  const leftFace = landmarks[234];
  const rightFace = landmarks[454];
  const noseBridge = landmarks[168];
  const noseBottom = landmarks[2];
  const noseLeft = landmarks[98];
  const noseRight = landmarks[327];
  const mouthLeft = landmarks[61];
  const mouthRight = landmarks[291];
  const leftEyeOuter = landmarks[33];
  const leftEyeInner = landmarks[133];
  const rightEyeInner = landmarks[362];
  const rightEyeOuter = landmarks[263];
  const leftEyeTop = landmarks[159];
  const leftEyeBottom = landmarks[145];
  const rightEyeTop = landmarks[386];
  const rightEyeBottom = landmarks[374];

  const browY = average([
    landmarks[70].y, landmarks[63].y, landmarks[105].y, landmarks[66].y, landmarks[107].y,
    landmarks[336].y, landmarks[296].y, landmarks[334].y, landmarks[293].y, landmarks[300].y
  ]);

  const faceWidth = distance(leftFace, rightFace);
  const faceLength = distance(faceTop, chin);
  const faceRatio = faceLength / faceWidth;
  const upperThird = Math.abs(browY - faceTop.y) / faceLength;
  const middleThird = Math.abs(noseBottom.y - browY) / faceLength;
  const lowerThird = Math.abs(chin.y - noseBottom.y) / faceLength;
  const eyeSpacing = distance(leftEyeInner, rightEyeInner) / faceWidth;
  const leftEyeSize = distance(leftEyeOuter, leftEyeInner) * distance(leftEyeTop, leftEyeBottom);
  const rightEyeSize = distance(rightEyeInner, rightEyeOuter) * distance(rightEyeTop, rightEyeBottom);
  const eyeSizeRatio = Math.min(leftEyeSize, rightEyeSize) / Math.max(leftEyeSize, rightEyeSize);
  const noseWidth = distance(noseLeft, noseRight) / faceWidth;
  const noseLength = distance(noseBridge, noseBottom) / faceLength;
  const mouthWidth = distance(mouthLeft, mouthRight) / faceWidth;
  const chinLength = distance(noseBottom, chin) / faceLength;
  const faceCenterX = (leftFace.x + rightFace.x) / 2;
  const featureCenterX = average([faceTop.x, chin.x, noseBridge.x, noseBottom.x, landmarks[13].x, landmarks[14].x]);
  const centerOffset = Math.abs(featureCenterX - faceCenterX) / faceWidth;
  const leftWidth = Math.abs(featureCenterX - leftFace.x);
  const rightWidth = Math.abs(rightFace.x - featureCenterX);
  const widthSymmetry = 1 - Math.abs(leftWidth - rightWidth) / Math.max(leftWidth, rightWidth);
  const overallSymmetry = clamp((widthSymmetry * .62 + eyeSizeRatio * .38) - centerOffset * .42, 0, 1);
  const thirdsSpread = Math.max(upperThird, middleThird, lowerThird) - Math.min(upperThird, middleThird, lowerThird);
  const thirdsBalance = clamp(1 - thirdsSpread * 2.8, 0, 1);
  const featureFocus = clamp(1 - Math.abs(eyeSpacing - .22) - Math.abs(noseWidth - .24) - Math.abs(mouthWidth - .39), 0, 1);

  const metrics = {
    faceWidth,
    faceLength,
    faceRatio,
    upperThird,
    middleThird,
    lowerThird,
    eyeSpacing,
    eyeSizeRatio,
    noseWidth,
    noseLength,
    mouthWidth,
    chinLength,
    widthSymmetry,
    centerOffset,
    overallSymmetry,
    thirdsBalance,
    featureFocus
  };

  return {
    ...metrics,
    archetype: classifyArchetype(metrics)
  };
}

function classifyArchetype(metrics) {
  const { faceRatio, lowerThird, noseWidth, mouthWidth, eyeSpacing, featureFocus, overallSymmetry } = metrics;

  if (faceRatio >= 1.5) {
    return {
      name: "马脸",
      tag: "马脸 · 清瘦理性型",
      desc: "脸长感明显，视觉上更成熟、冷静，适合走利落、专业、有距离感的高级路线。",
      hair: "推荐中长层次、侧分刘海或带蓬松度的发型来缩短脸部纵向感；避雷贴头直发和过高颅顶，会让脸显得更长。"
    };
  }

  if (faceRatio <= 1.16 && lowerThird < .37) {
    return {
      name: "娃娃脸",
      tag: "娃娃脸 · 亲和减龄型",
      desc: "面部圆润度和亲和感更强，第一印象偏柔和、年轻，容易让人降低防备。",
      hair: "推荐空气感刘海、锁骨发或柔和卷度；避雷厚重齐刘海和过圆短发，否则会加重幼态感。"
    };
  }

  if (faceRatio <= 1.24 && lowerThird >= .36) {
    return {
      name: "方圆脸",
      tag: "方圆脸 · 稳重可靠型",
      desc: "下庭稳定、骨相承托感较强，气质上更踏实、耐看，适合做可靠型个人形象。",
      hair: "推荐八字刘海、侧分层次和下颌线外翻层次来柔化轮廓；避雷一刀切齐短发和贴脸中分。"
    };
  }

  if (noseWidth >= .28 && mouthWidth <= .36) {
    return {
      name: "菱形脸",
      tag: "菱形脸 · 锋芒辨识型",
      desc: "中部轮廓存在识别点，气质更有锋芒和记忆度，适合打造精致、清冷或艺术感路线。",
      hair: "推荐蓬松太阳穴区域、法式刘海或耳侧层次来平衡中面部；避雷贴头高马尾和露全脸紧发。"
    };
  }

  if (eyeSpacing <= .2 && featureFocus >= .72) {
    return {
      name: "狐狸脸",
      tag: "狐狸脸 · 精明灵动型",
      desc: "五官集中度较高，眉眼存在灵动感，容易给人聪明、反应快、会抓重点的印象。",
      hair: "推荐侧分、轻薄长刘海和带线条感的层次发，突出眉眼优势；避雷过厚遮眼刘海，会削弱灵动感。"
    };
  }

  if (mouthWidth >= .46) {
    return {
      name: "鲶鱼脸",
      tag: "鲶鱼脸 · 松弛表达型",
      desc: "口部表达感较强，整体气质更有松弛和亲近感，适合自然、生活化、带笑意的形象表达。",
      hair: "推荐自然卷度、低层次长发或慵懒感造型；避雷过于锋利的短直线条，会和松弛气质冲突。"
    };
  }

  if (faceRatio >= 1.26 && faceRatio <= 1.44 && overallSymmetry >= .78) {
    return {
      name: "鹅蛋脸",
      tag: "鹅蛋脸 · 均衡耐看型",
      desc: "脸部比例较均衡，轮廓柔和但不松散，是大众审美里比较耐看、适配度高的脸型。",
      hair: "大多数发型都能驾驭，推荐低层次锁骨发、侧分或自然中分；避雷过度遮脸，反而浪费比例优势。"
    };
  }

  return {
    name: "自然脸型",
    tag: "自然脸型 · 清爽协调型",
    desc: "整体轮廓没有过强偏向，适合通过发型、光线和穿搭建立个人记忆点。",
    hair: "推荐根据当天状态选择清爽露额或轻侧分造型；避雷贴头、油腻和过度堆叠的复杂发型。"
  };
}

function scoreFromMetrics(metrics) {
  const thirds = metrics.thirdsBalance * 28;
  const symmetry = metrics.overallSymmetry * 30;
  const focus = metrics.featureFocus * 18;
  const faceRatioScore = (1 - Math.min(Math.abs(metrics.faceRatio - 1.34) / .38, 1)) * 14;
  const eyeScore = metrics.eyeSizeRatio * 10;
  return Math.round(clamp(42 + thirds + symmetry + focus + faceRatioScore + eyeScore - 34, 52, 96));
}

function isSkinLikePixel(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
  const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;

  return r > 38 && g > 25 && b > 16 && max - min > 8 && cr > 128 && cr < 184 && cb > 72 && cb < 150;
}

function fallbackPoint(x, y, z = 0) {
  return {
    x: clamp(x, 0, 1),
    y: clamp(y, 0, 1),
    z
  };
}

function estimateFallbackFaceBox(image) {
  const canvas = document.createElement("canvas");
  const maxSide = 360;
  const scale = Math.min(maxSide / image.width, maxSide / image.height, 1);
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);
  const step = 3;
  const leftLimit = width * .08;
  const rightLimit = width * .92;
  const topLimit = height * .06;
  const bottomLimit = height * .94;

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let count = 0;

  for (let y = Math.floor(topLimit); y < bottomLimit; y += step) {
    for (let x = Math.floor(leftLimit); x < rightLimit; x += step) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];

      if (!isSkinLikePixel(r, g, b)) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      count += 1;
    }
  }

  const sampled = Math.max(1, ((rightLimit - leftLimit) / step) * ((bottomLimit - topLimit) / step));
  const skinRatio = count / sampled;
  const lowConfidence = skinRatio < .012;

  if (image.width < 120 || image.height < 120) return null;

  if (lowConfidence) {
    return {
      x: .22,
      y: .10,
      w: .56,
      h: .78,
      confidence: "low"
    };
  }

  let x = minX / width;
  let y = minY / height;
  let w = (maxX - minX) / width;
  let h = (maxY - minY) / height;
  const centerX = x + w / 2;
  const expandedW = clamp(w * 1.22, .36, .72);
  let expandedH = clamp(h * 1.34, expandedW * 1.18, expandedW * 1.68);
  const expandedY = y - expandedH * .12;

  if (expandedY + expandedH > .98) {
    expandedH = .98 - Math.max(.02, expandedY);
  }

  return {
    x: clamp(centerX - expandedW / 2, .02, .98 - expandedW),
    y: clamp(expandedY, .02, .98 - expandedH),
    w: expandedW,
    h: expandedH,
    confidence: skinRatio > .04 ? "medium" : "low"
  };
}

function buildFallbackLandmarks(box) {
  const landmarks = Array.from({ length: 478 }, () => fallbackPoint(box.x + box.w / 2, box.y + box.h / 2));
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h * .53;
  const rx = box.w * .45;
  const ry = box.h * .48;

  const set = (index, x, y) => {
    landmarks[index] = fallbackPoint(x, y);
  };

  const contour = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454];
  contour.forEach((index, order) => {
    const t = order / (contour.length - 1);
    const angle = Math.PI - t * Math.PI;
    set(index, cx + Math.cos(angle) * rx, cy + Math.sin(angle) * ry);
  });

  const topY = box.y + box.h * .05;
  const browY = box.y + box.h * .30;
  const eyeY = box.y + box.h * .38;
  const noseBridgeY = box.y + box.h * .44;
  const noseBottomY = box.y + box.h * .61;
  const mouthY = box.y + box.h * .74;
  const chinY = box.y + box.h * .98;
  const eyeOuterGap = box.w * .34;
  const eyeInnerGap = box.w * .12;
  const eyeHalfHeight = box.h * .018;
  const noseHalfWidth = box.w * .105;
  const mouthHalfWidth = box.w * .205;

  set(10, cx, topY);
  set(152, cx, chinY);
  set(168, cx, noseBridgeY);
  set(2, cx, noseBottomY);
  set(98, cx - noseHalfWidth, noseBottomY);
  set(327, cx + noseHalfWidth, noseBottomY);
  set(61, cx - mouthHalfWidth, mouthY);
  set(291, cx + mouthHalfWidth, mouthY);
  set(13, cx, mouthY - box.h * .025);
  set(14, cx, mouthY + box.h * .035);

  set(33, cx - eyeOuterGap, eyeY);
  set(133, cx - eyeInnerGap, eyeY);
  set(362, cx + eyeInnerGap, eyeY);
  set(263, cx + eyeOuterGap, eyeY);
  set(159, cx - box.w * .23, eyeY - eyeHalfHeight);
  set(145, cx - box.w * .23, eyeY + eyeHalfHeight);
  set(386, cx + box.w * .23, eyeY - eyeHalfHeight);
  set(374, cx + box.w * .23, eyeY + eyeHalfHeight);

  [70, 63, 105, 66, 107].forEach((index, order) => {
    set(index, cx - box.w * (.31 - order * .04), browY + (order % 2 ? -box.h * .01 : 0));
  });
  [336, 296, 334, 293, 300].forEach((index, order) => {
    set(index, cx + box.w * (.15 + order * .04), browY + (order % 2 ? -box.h * .01 : 0));
  });

  return landmarks;
}

function createFallbackDetection(image) {
  const box = estimateFallbackFaceBox(image);
  if (!box) return null;

  return {
    landmarks: buildFallbackLandmarks(box),
    confidence: box.confidence
  };
}

function drawOverlay(landmarks, image) {
  const canvas = refs.canvas;
  const context = canvas.getContext("2d");
  const size = refs.dropZone.getBoundingClientRect().width;
  const scale = Math.max(size / image.width, size / image.height);
  const drawWidth = image.width * scale;
  const drawHeight = image.height * scale;
  const offsetX = (size - drawWidth) / 2;
  const offsetY = (size - drawHeight) / 2;

  canvas.width = Math.round(size * window.devicePixelRatio);
  canvas.height = Math.round(size * window.devicePixelRatio);
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  context.scale(window.devicePixelRatio, window.devicePixelRatio);
  context.clearRect(0, 0, size, size);

  const toCanvas = (point) => ({
    x: offsetX + point.x * drawWidth,
    y: offsetY + point.y * drawHeight
  });

  const drawLine = (indices, color = "rgba(255,255,255,.82)", width = 1.4) => {
    context.beginPath();
    indices.forEach((index, order) => {
      const point = toCanvas(landmarks[index]);
      if (order === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.strokeStyle = color;
    context.lineWidth = width;
    context.stroke();
  };

  drawLine([10, 168, 2, 13, 14, 152], "rgba(255,255,255,.92)", 1.5);
  drawLine([234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454], "rgba(255,255,255,.72)", 1.2);
  drawLine([33, 133, 362, 263], "rgba(110,107,255,.92)", 1.5);
  drawLine([98, 2, 327], "rgba(167,139,250,.9)", 1.4);
  drawLine([61, 13, 291, 14, 61], "rgba(240,185,198,.86)", 1.2);

  [10, 152, 234, 454, 33, 133, 362, 263, 98, 327, 61, 291, 168, 2].forEach((index) => {
    const point = toCanvas(landmarks[index]);
    context.beginPath();
    context.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
    context.fillStyle = "rgba(255,255,255,.96)";
    context.fill();
    context.lineWidth = 1.5;
    context.strokeStyle = "rgba(110,107,255,.78)";
    context.stroke();
  });
}

function clearOverlay() {
  const canvas = refs.canvas;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function updateRatio(id, value, min, max, formatter = percent) {
  $(id).textContent = formatter(value);
  $(`${id}Bar`).style.setProperty("--bar", barPercent(value, min, max));
}

function ratioSentence(kind, value) {
  if (kind === "upper") {
    if (value > .36) return "额头感更开阔，规划感较强";
    if (value < .27) return "上庭偏紧凑，适合清爽露额";
    return "上庭比例自然，观感较稳";
  }

  if (kind === "middle") {
    if (value > .38) return "中庭承接感明显，执行气质较强";
    if (value < .29) return "中庭偏短，亲和度更高";
    return "中庭比例协调，五官衔接顺";
  }

  if (kind === "lower") {
    if (value > .39) return "下庭稳定，耐看度较强";
    if (value < .29) return "下庭偏轻，建议增强轮廓感";
    return "下庭比例适中，气质平衡";
  }

  if (value < .2) return "五官更集中，眉眼记忆点强";
  if (value > .3) return "五官更舒展，亲和留白足";
  return "五眼比例适中，视觉较舒服";
}

function updateMetricsUI(metrics, total) {
  $("scoreNumber").textContent = total;
  $("scoreRing").style.setProperty("--score-angle", `${Math.round(total / 100 * 360)}deg`);
  $("photoState").textContent = "识别完成";
  $("faceType").textContent = metrics.archetype.tag;
  $("archetypeTag").textContent = metrics.archetype.name;
  $("faceDesc").textContent = metrics.archetype.desc;
  $("hairText").textContent = metrics.archetype.hair;

  $("upperInsight").textContent = ratioSentence("upper", metrics.upperThird);
  $("middleInsight").textContent = ratioSentence("middle", metrics.middleThird);
  $("lowerInsight").textContent = ratioSentence("lower", metrics.lowerThird);
  $("eyeInsight").textContent = ratioSentence("eye", metrics.eyeSpacing);

  updateRatio("upperThird", metrics.upperThird, .22, .42);
  updateRatio("middleThird", metrics.middleThird, .24, .45);
  updateRatio("lowerThird", metrics.lowerThird, .24, .45);
  updateRatio("eyeSpacing", metrics.eyeSpacing, .16, .34, (value) => value.toFixed(2));
}

function starString(score) {
  return "★★★★★".slice(0, score) + "☆☆☆☆☆".slice(0, 5 - score);
}

function scoreToFive(value) {
  return clamp(Math.round(value), 1, 5);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeXml(value) {
  return escapeHtml(value).replaceAll("'", "&apos;");
}

function observationPhrase(metrics) {
  const balance = metrics.thirdsBalance >= .78 ? "三庭衔接较顺" : "三庭比例存在轻微偏差";
  const symmetry = metrics.overallSymmetry >= .82 ? "左右平衡感较好" : "左右平衡仍有可见差异";
  const focus = metrics.featureFocus >= .76 ? "五官聚合感强" : metrics.featureFocus >= .55 ? "五官分布适中" : "五官留白感更明显";
  return `${balance}，${symmetry}，${focus}`;
}

function ageDescription() {
  const value = $("ageInput").value.trim();
  if (value) return `用户提供年龄为 ${value} 岁，以下会结合该年龄阶段作传统文化角度的倾向分析。`;
  return "用户未提供年龄；本工具只能根据面部成熟度、骨相轮廓与神态作外观区间推测，倾向青年至成熟青年区间，谨慎判断。";
}

function buildDimension(score, evidence, tendency, advantage, weakness, direction, risk) {
  const numeric = scoreToFive(score);
  return {
    score: numeric,
    stars: starString(numeric),
    evidence,
    tendency,
    advantage,
    weakness,
    direction,
    risk,
    analysis: `面部依据：${evidence} 相法结论：${tendency}${advantage} 现实建议：${direction} 需注意：${risk}`
  };
}

function dimensionLong(title, dimension, extra) {
  return `### ${title}：${dimension.stars}

先看面部依据，${dimension.evidence}${extra || ""} 从传统相法的语言来说，这一维不是看某一个点就下结论，而是看神、气、形之间是否能互相承接。所谓“清”，是眼神和五官不散；所谓“正”，是中轴和轮廓不乱；所谓“收”，是气质不外泄、不浮躁。由此推断，${dimension.tendency}${dimension.advantage}

落到现实生活里，${dimension.direction}${dimension.weakness} 这里不作绝对宿命判断，更像人生中容易反复出现的一种倾向：当状态稳定、方向明确时，这一维更容易出成绩；当情绪被外界带走、节奏被打乱时，短板就会放大。评分给到 ${dimension.stars}，原因是可见结构里有支撑点，也有需要靠后天习惯修正的地方。${dimension.risk}`;
}

function buildLongSections(report, metrics, context) {
  const { thirds, symmetry, focus, upper, middle, lower, archetype, temperament } = context;
  const scores = report.scores;
  const age = report.basic_info.estimated_age_range;

  return {
    disclaimer: "说明：以下属于传统面相文化角度的娱乐性解读，不等同于现实中的性格鉴定、命运判断或医学、法律、投资建议。照片有光线、角度、美颜或遮挡影响，部分细节可能不完全清晰，因此以“倾向分析”为主。",
    basic: `从照片可见信息判断，${age}本报告不识别真实身份，也不声称知道用户的真实职业、收入或家庭背景，只依据照片中可见的面部结构进行传统文化角度的观察。整体脸型被归入「${archetype.tag}」，这是一种比较容易被大众理解的脸型类比，并不是严格的人种、身份或命运标签。脸长与脸宽比例约为 ${metrics.faceRatio.toFixed(2)}，三庭比例约为 ${thirds}，整体对称度约为 ${symmetry}%，五官状态呈现为${focus}。

具体来看，额头部分因发际线、刘海、光线可能影响判断，因此只按上庭估算处理：${upper}。眉眼区域是本相最需要看的地方，眼部左右大小接近度约 ${Math.round(metrics.eyeSizeRatio * 100)}%，眼距与脸宽比例约 ${metrics.eyeSpacing.toFixed(2)}，说明神态与五眼结构有一定可读性。鼻梁、鼻翼和鼻准在本模型中以鼻宽、鼻长比例估算，鼻宽/脸宽约 ${metrics.noseWidth.toFixed(2)}，鼻长/脸长约 ${metrics.noseLength.toFixed(2)}。口唇部分以嘴宽和闭合结构为主，嘴宽/脸宽约 ${metrics.mouthWidth.toFixed(2)}。下巴与下颌用于观察承托感，下庭比例约 ${percent(metrics.lowerThird)}。耳朵、人中、法令若被角度、发型或清晰度影响，均需写作照片中不够清晰，谨慎判断。整体第一印象是${temperament}，不属于一眼极端张扬的相，更偏向需要从结构、神态和长期气质里慢慢读出的类型。`,
    basicAnalysis: `此相最明显的关键词，可以概括为“${metrics.featureFocus >= .76 ? "清、灵、收" : metrics.overallSymmetry >= .82 ? "清、正、稳" : "稳、静、藏"}”。所谓清，指五官之间有可辨识的边界，不至于杂乱无章；所谓正，指面部中轴、左右平衡和眼鼻口承接关系有一定秩序；所谓收，指神态不外散，气质不浮。传统相法里，眉主情义与行动力，眼主心神与判断力，鼻主财帛与执行力，口主表达与承诺，下巴主晚运与承托。放在此相中看，眉眼给出的信息是“先观察、后出手”，鼻口结构给出的信息是“适合正道积累，不宜靠一时冒进”，下庭给出的信息是“越稳定越显后劲”。

从整体格局看，${report.core_result.summary} 这类面相若用老师傅的话说，不贵在一开始锋芒外露，而贵在气能收、形能定。也就是说，真正的关键不在短时间内把自己推到最热闹的位置，而在长期里让别人看见可靠、专业和可持续。若三庭衔接顺，现实中往往代表做事有章法；若五官集中，容易有记忆点和判断力；若下庭稳定，则在关系、家庭、事业后半程更容易积累信任。需要提醒的是，照片只是一刻状态，光线、表情、睡眠和拍摄角度都会改变观感，所以此处所有判断都以倾向为主。`,
    personality: `从神态、眉眼、鼻相、口相、脸型和整体气质综合看，你更像是先观察、再行动的人。你未必是一见面就特别外放的类型，表面看似平静，内在其实会处理很多信息：对方说话的分寸、环境是否安全、事情有没有长期价值，这些都会影响你的判断。眉眼结构显示你对细节较敏感，优点是能看见别人忽略的问题，短板是容易想得多、行动慢；鼻口结构显示你更适合靠稳定输出获得认可，而不是靠情绪化表达获得短暂注意。

在人际模式上，你更适合“少而精”的关系。你不是不能热闹，而是不适合长时间处在消耗型社交里。真正能滋养你的关系，是对方讲信用、有边界、能一起做事，也能给彼此空间。情绪处理上，你容易先压住，再自己消化；这在短期里显得成熟，但长期容易形成内耗。成长关键在于把观察力转成行动力，把判断力转成表达力。此相之贵，不在张扬，而在后劲；你的优势不是冲得最快，而是能复盘、能沉淀、能在长期里慢慢见成色。`,
    dimensions: [
      dimensionLong("事业", scores.career, "眉眼主判断，鼻主承担，颧颊主责任感；当前结构显示事业气更适合从专业可信度里生出来。"),
      dimensionLong("财运", scores.wealth, "鼻相、口相和气色共同影响财帛观感；此处只作传统文化角度观察，不断言财富结果。"),
      dimensionLong("婚姻", scores.love, "眉眼距离感、口唇表达感和下巴承托感，会影响亲密关系里的表达方式与稳定感。"),
      dimensionLong("生命", scores.vitality, "这里不能做医学判断，只从传统面相中的精气神、气色、眼神稳定度和生活状态倾向分析。"),
      dimensionLong("子女", scores.children_family, "眼下、卧蚕、人中、法令在照片中未必清晰，因此本项谨慎判断，不断言具体子女数量。"),
      dimensionLong("智慧", scores.wisdom, "额头、眉眼、五官协调度与神态稳定感共同显示学习力、判断力和复盘能力。")
    ].join("\n\n"),
    persona: `${report.persona_label.label}

这个标签适合你，是因为它不是单看脸型得出的口号，而是来自六维结果的交叉判断。事业和智慧维度说明你适合靠专业、复盘、判断力慢慢打开局面；感情和家庭维度说明你需要稳定、具体、可兑现的关系；生命状态维度则提醒你，精神气色对你的外在气韵影响很明显。换句话说，你不是那种靠夸张表达赢得注意的人，更像是越稳定、越自律、越有作品，越能显出价值的人。`,
    forecast: `未来一年更像是调整方向、积累筹码、关系筛选的年份。事业上，关键词是定方向，不要什么都想做；财运上，关键词是稳财务，把现金流和长期技能放在第一位；感情上，关键词是练表达，不要让重要的人一直猜；人际上，关键词是筛圈层，远离只消耗情绪的人；自我成长上，关键词是抓小机会，把每一次小输出都变成可见成果。

从传统相法角度看，未来一年不一定是马上飞跃的年份，但很适合把基础打稳。真正需要避开的风险，不是某个神秘灾难，而是长期拖延、长期熬夜、长期压抑、长期不表达。如果你能把节奏稳住，把形象清爽化，把一个核心能力做成标签，未来一年会逐渐看清真正适合自己的路。`,
    challenges: report.life_challenges.map((item, index) => `第${index + 1}类是${item.title}。产生原因：${item.reason}面相依据在于眉眼与中轴结构会让你先看见风险，再决定是否行动。现实表现：${item.manifestation}化解建议：${item.advice}`).join("\n\n"),
    blessings: report.life_blessings.map((item, index) => `第${index + 1}类是${item.title}。福报来源：${item.source}面相依据在于结构不散、气质可收，容易通过长期稳定获得信任。触发条件：${item.trigger_condition}如何放大：${item.amplify_method}`).join("\n\n"),
    finalAdvice: `事业方面，${report.final_advice.career} 财运方面，${report.final_advice.wealth} 感情方面，${report.final_advice.love} 生活作息方面，${report.final_advice.lifestyle} 人际关系方面，${report.final_advice.relationships} 自我成长方面，${report.final_advice.self_growth}

总体看，你属于越稳定越有运、越自律越显气质的类型。别急着成为别人眼中的热闹人物，先成为一个真正有底气、有技能、有选择权的人。传统相法讲“形正则气顺，气顺则神安”，放到现代生活里，就是让自己的作息、表达、能力和关系都慢慢归位。`
  };
}

function buildStructuredReport(metrics, total) {
  const archetype = metrics.archetype;
  const thirds = `${Math.round(metrics.upperThird * 100)}:${Math.round(metrics.middleThird * 100)}:${Math.round(metrics.lowerThird * 100)}`;
  const symmetry = Math.round(metrics.overallSymmetry * 100);
  const focus = metrics.featureFocus >= .76 ? "五官集中、记忆点强" : metrics.featureFocus >= .55 ? "五官分布适中、观感舒服" : "五官舒展、留白感强";
  const upper = ratioSentence("upper", metrics.upperThird);
  const middle = ratioSentence("middle", metrics.middleThird);
  const lower = ratioSentence("lower", metrics.lowerThird);
  const temperament = metrics.overallSymmetry >= .82 && metrics.thirdsBalance >= .78
    ? "清秀稳进型"
    : metrics.featureFocus >= .76
      ? "锋芒灵动型"
      : "温和后发型";
  const keywords = [archetype.name, focus, `${symmetry}%对称度`, temperament];

  const careerScore = 2.4 + metrics.thirdsBalance * 1.4 + metrics.overallSymmetry * .9 + metrics.noseLength * 1.2;
  const wealthScore = 2.2 + metrics.noseWidth * 5.2 + metrics.mouthWidth * 1.4 + metrics.thirdsBalance * .8;
  const loveScore = 2.1 + metrics.overallSymmetry * 1.2 + metrics.mouthWidth * 2.4 + (metrics.archetype.name === "娃娃脸" ? .5 : 0);
  const vitalityScore = 2.2 + metrics.eyeSizeRatio * 1.2 + metrics.overallSymmetry * .9 + metrics.thirdsBalance * .7;
  const familyScore = 2.1 + metrics.lowerThird * 5.2 + metrics.mouthWidth * 1.1 + metrics.overallSymmetry * .6;
  const wisdomScore = 2.2 + metrics.upperThird * 4.5 + metrics.eyeSizeRatio * 1.1 + metrics.featureFocus * .8;

  const scores = {
    career: buildDimension(
      careerScore,
      `${middle}，面部中轴线偏移约 ${metrics.centerOffset.toFixed(2)}，承接感可读。`,
      "事业上更适合先建立专业可信度，再逐步放大影响力。",
      "做事不宜靠一时冲动，越是长期项目越能体现后劲。",
      "若急于求成，容易把精力分散在多个方向。",
      "适合项目推进、咨询服务、产品运营、专业交付和需要稳定判断的岗位。",
      "避开短线投机和频繁换赛道，先把一个标签做深。"
    ),
    wealth: buildDimension(
      wealthScore,
      `鼻宽/脸宽约 ${metrics.noseWidth.toFixed(2)}，嘴宽/脸宽约 ${metrics.mouthWidth.toFixed(2)}。`,
      "财运更偏正财与长期积累，适合通过稳定技能和口碑变现。",
      "有持续经营意识，适合把资源留在可复利的事情上。",
      "遇到高收益诱惑时，容易被短期情绪影响判断。",
      "建议做预算、分账户储蓄，把收入来源从单一劳动逐步过渡到作品、服务或长期客户。",
      "少碰看不懂的投资，避免人情借贷。"
    ),
    love: buildDimension(
      loveScore,
      `口相表达感与下庭承托感共同作用，脸型为「${archetype.tag}」。`,
      "感情里更需要被理解和被稳定回应，不适合长期猜测式沟通。",
      "相处久了会显出可靠和细腻，适合慢热但认真经营的关系。",
      "情绪不说清时，容易让对方误判你的真实需求。",
      "建议把期待说具体：时间、陪伴、边界和承诺都越清楚越好。",
      "避开只靠新鲜感推动的关系。"
    ),
    vitality: buildDimension(
      vitalityScore,
      `眼部左右大小接近度约 ${Math.round(metrics.eyeSizeRatio * 100)}%，整体对称度约 ${symmetry}%。`,
      "传统望诊文化中，神采稳定代表精神状态更容易被看见。",
      "只要睡眠和光线状态好，整个人会显得更清爽有气。",
      "熬夜、疲惫、低光照片会明显拉低气色观感。",
      "建议保持规律作息、自然光拍照、适度运动，让眼神先亮起来。",
      "本项只作传统文化参考，不作医学判断。"
    ),
    children_family: buildDimension(
      familyScore,
      `下庭比例约 ${percent(metrics.lowerThird)}，嘴部闭合和下巴承托共同影响亲缘观感。`,
      "家庭缘分更看重陪伴质量与稳定情绪，而不是外在热闹。",
      "适合做长期经营型关系，越稳定越容易积累信任。",
      "若压力大时沉默太久，家人容易感到距离。",
      "建议把关心落到具体行动，比如固定沟通、共同计划和实际照顾。",
      "不要用面相断言子女数量，本报告只分析家庭经营倾向。"
    ),
    wisdom: buildDimension(
      wisdomScore,
      `${upper}，眉眼五官集中度为 ${Math.round(metrics.featureFocus * 100)}%。`,
      "学习力和判断力偏向观察后再出手，适合复盘型成长。",
      "能从细节里抓重点，适合需要洞察、审美、策略的任务。",
      "想太多时容易拖延行动。",
      "建议用小步验证代替长时间内耗，每周固定复盘一次。",
      "避免过度追求完美而错过窗口。"
    )
  };

  const persona = metrics.archetype.name === "狐狸脸"
    ? "眉眼灵动、后劲很强的观察型行动者"
    : metrics.archetype.name === "方圆脸"
      ? "外稳内韧、财运稳进的长期主义者"
      : metrics.archetype.name === "娃娃脸"
        ? "亲和柔软、越相处越有吸引力的慢热型人格"
        : "清爽克制、靠专业感打开局面的稳进型人格";

  const report = {
    basic_info: {
      estimated_gender: "不做身份或性别断定，仅观察照片中的气质呈现",
      estimated_age_range: ageDescription(),
      face_shape: archetype.tag,
      overall_temperament: temperament,
      keywords
    },
    face_metrics: {
      forehead: upper,
      eyebrows: "眉眼区域用于估算上庭和神采，若有刘海或阴影需谨慎判断。",
      eyes: `眼部接近度约 ${Math.round(metrics.eyeSizeRatio * 100)}%，${ratioSentence("eye", metrics.eyeSpacing)}。`,
      nose: `鼻宽/脸宽约 ${metrics.noseWidth.toFixed(2)}，鼻长/脸长约 ${metrics.noseLength.toFixed(2)}。`,
      cheeks: `脸部左右宽度平衡约 ${Math.round(metrics.widthSymmetry * 100)}%，颧部受光线和镜头影响较大。`,
      ears: "照片中耳部通常受发型和角度遮挡，谨慎判断。",
      philtrum_and_nasolabial: "人中与法令在当前模型中不做强断，只作口鼻区域辅助观察。",
      mouth: `嘴宽/脸宽约 ${metrics.mouthWidth.toFixed(2)}，表达感用于判断沟通风格倾向。`,
      jaw_and_chin: `${lower}，下巴长度/脸长约 ${metrics.chinLength.toFixed(2)}。`,
      hairline: "发际线可能被刘海遮挡，上庭为额头估算，仅作参考。",
      complexion: "气色只按照片光线下的精神观感表达，不作医学诊断。",
      photo_limitations: "照片角度、光线、滤镜、美颜、镜头畸变都会影响比例判断。"
    },
    core_result: {
      archetype: archetype.tag,
      summary: `此相之贵，不在张扬，而在结构可读。${observationPhrase(metrics)}，因此更适合以清爽、稳定、可信的方式建立个人气质。`,
      confidence: Math.round((metrics.thirdsBalance * .35 + metrics.overallSymmetry * .35 + metrics.featureFocus * .3) * 100)
    },
    scores,
    persona_label: {
      label: persona,
      reason: `这个标签来自「${archetype.tag}」、三庭比例 ${thirds}、对称度 ${symmetry}% 和五官集中度 ${Math.round(metrics.featureFocus * 100)}%。它不是命运断语，而是把照片中可见结构转译为更容易理解的人设气质。`
    },
    one_year_forecast: {
      career: "未来一年事业上适合先做减法，把一个核心能力做成可被识别的标签，再争取更大的舞台。",
      wealth: "财务状态宜稳不宜赌，正财、长期客户、复购型收入比一时偏财更适合你。",
      love: "感情上更适合慢热确认，不宜用沉默测试对方。把需求说清楚，会减少误会。",
      relationships: "人际关系里贵人多来自专业场景和旧关系复联，少参加低质量热闹。",
      growth: "自我成长关键词是复盘、稳定输出、形象清爽化。",
      risk_warning: "需要避开情绪内耗、短期诱惑和为了迎合别人而频繁改变方向。"
    },
    life_challenges: [
      {
        title: "情绪内耗关",
        reason: "眉眼观察力强，容易先看见细节和风险。",
        manifestation: "现实中可能表现为反复比较、迟迟不行动。",
        advice: "用小步试错替代长时间想象，先完成一个可验证动作。"
      },
      {
        title: "事业选择关",
        reason: "中庭承接感需要稳定方向来发挥。",
        manifestation: "机会多时容易分散，主线不够清晰。",
        advice: "一年只押一个主标签，其他机会作为辅助。"
      },
      {
        title: "感情沟通关",
        reason: "口相表达感需要具体表达才能被接住。",
        manifestation: "想被理解，但不一定把话说透。",
        advice: "把期待拆成具体行为，少用暗示，多用清楚表达。"
      }
    ],
    life_blessings: [
      {
        title: "后发之运",
        source: "三庭结构和对称度给人长期稳定的观感。",
        trigger_condition: "持续在同一方向积累作品、客户或口碑。",
        amplify_method: "少换赛道，多做复盘，把稳定变成品牌感。"
      },
      {
        title: "贵人助力",
        source: "五官协调度带来较好的第一印象和信任感。",
        trigger_condition: "在专业场合保持清爽表达和可靠交付。",
        amplify_method: "主动维护旧关系，及时反馈进展。"
      },
      {
        title: "智慧成长",
        source: "眉眼与上庭结构显示观察和判断优势。",
        trigger_condition: "把观察力用于学习、策略和表达，而不是内耗。",
        amplify_method: "建立固定输入输出节奏，形成可见成果。"
      }
    ],
    final_advice: {
      career: "事业上先定主线，再谈扩张；适合用作品、案例和长期交付建立权威。",
      wealth: "财运上重视现金流和复利积累，少碰看不懂的高波动机会。",
      love: "感情里表达越具体越有安全感，慢热没有问题，但不要让对方一直猜。",
      lifestyle: "生活方式上重视睡眠、自然光和运动，精神状态会明显提升面部气韵。",
      relationships: "人际上少而精，选择能互相成就的圈层。",
      self_growth: "自我提升上把观察力落成行动力，每周输出一次复盘或作品。"
    },
    disclaimer: "本报告基于传统面相文化与照片可见特征生成，仅供娱乐和文化参考，不构成医学、法律、投资、婚恋等专业建议。"
  };

  report.long_sections = buildLongSections(report, metrics, { thirds, symmetry, focus, upper, middle, lower, archetype, temperament });

  return {
    title: `${archetype.name}识别完成，气韵分 ${total}`,
    verdict: `此相之贵，不在张扬，而在后劲；${focus}，宜走长期积累之路。`,
    summary: `${metrics.detectionMode === "estimate" ? "当前为本地结构估算模式，细节结果以倾向参考为主。 " : ""}先看结构：${observationPhrase(metrics)}。再看相法：你的趣味脸型结论为「${archetype.tag}」，三庭比例约 ${thirds}，整体对称度约 ${symmetry}%。这不是单纯读数，而是把可见面部依据转译为更容易理解的现实建议。`,
    romance: shortCard(report.scores.love),
    wealth: `事业：${shortCard(report.scores.career)}\n\n财运：${shortCard(report.scores.wealth)}`,
    talent: shortCard(report.scores.wisdom),
    health: shortCard(report.scores.vitality),
    family: shortCard(report.scores.children_family),
    career: shortCard(report.scores.career),
    stars: {
      romance: report.scores.love.stars,
      wealth: report.scores.wealth.stars,
      talent: report.scores.wisdom.stars,
      health: report.scores.vitality.stars,
      family: report.scores.children_family.stars,
      career: report.scores.career.stars
    },
    full: report,
    fullText: buildMarkdownReport(report)
  };
}

function shortCard(dimension) {
  return `面部依据：${dimension.evidence} 相法解读：${dimension.tendency} 现实建议：${dimension.direction}`;
}

function buildMarkdownReport(report) {
  const s = report.long_sections;
  return `> ${s.disclaimer}

## 面相基本信息及构成

${s.basic}

## 面相基本分析

${s.basicAnalysis}

## 命理性格详细分析

${s.personality}

## 个性特点：六维结构化解读分析

${s.dimensions}

## 人设标签

${s.persona}

## 未来1年趋势与预测

${s.forecast}

## 一生将会遇到的劫难

${s.challenges}

## 一生将会遇到的福报

${s.blessings}

## 综合建议

${s.finalAdvice}`;
}

function renderFullReport(report) {
  const s = report.long_sections;
  const blocks = [
    ["说明", s.disclaimer],
    ["面相基本信息及构成", s.basic],
    ["面相基本分析", s.basicAnalysis],
    ["命理性格详细分析", s.personality],
    ["个性特点：六维结构化解读分析", s.dimensions],
    ["人设标签", s.persona],
    ["未来1年趋势与预测", s.forecast],
    ["一生将会遇到的劫难", s.challenges],
    ["一生将会遇到的福报", s.blessings],
    ["综合建议", s.finalAdvice]
  ];

  $("fullReportContent").innerHTML = `
    <section class="report-block">
      <h3>报告索引</h3>
      <div class="info-grid">
        <div><span>脸型判断</span><strong>${escapeHtml(report.basic_info.face_shape)}</strong></div>
        <div><span>整体气质</span><strong>${escapeHtml(report.basic_info.overall_temperament)}</strong></div>
        <div><span>面相关键词</span><strong>${escapeHtml(report.basic_info.keywords.join("、"))}</strong></div>
        <div><span>置信度</span><strong>${report.core_result.confidence}%</strong></div>
        <div><span>输出结构</span><strong>观察 · 推断 · 建议</strong></div>
      </div>
    </section>
    ${blocks.map(([title, content]) => `
      <section class="report-block">
        <h3>${escapeHtml(title)}</h3>
        ${escapeHtml(content).split("\n\n").map((paragraph) => {
          if (paragraph.startsWith("### ")) {
            const [heading, ...rest] = paragraph.split("\n");
            return `<h4>${escapeHtml(heading.replace("### ", ""))}</h4><p>${escapeHtml(rest.join("\n"))}</p>`;
          }
          return `<p>${paragraph.replace(/\n/g, "<br>")}</p>`;
        }).join("")}
      </section>
    `).join("")}
  `;
}

function renderReport(metrics) {
  const total = scoreFromMetrics(metrics);
  updateMetricsUI(metrics, total);
  const copy = buildStructuredReport(metrics, total);
  state.latestReport = { total, ...copy };

  $("reportTitle").textContent = copy.title;
  $("coreVerdict").textContent = copy.verdict;
  $("reportSummary").textContent = copy.summary;
  $("romanceStars").textContent = copy.stars.romance;
  $("wealthStars").textContent = copy.stars.wealth;
  $("talentStars").textContent = copy.stars.talent;
  $("healthStars").textContent = copy.stars.health;
  $("familyStars").textContent = copy.stars.family;
  $("careerStars").textContent = copy.stars.career;
  $("romanceText").textContent = copy.romance;
  $("wealthText").textContent = copy.wealth;
  $("talentText").textContent = copy.talent;
  $("healthText").textContent = copy.health;
  $("familyText").textContent = copy.family;
  $("careerText").textContent = copy.career;
  $("fullReportSection").hidden = true;
  renderFullReport(copy.full);
}

async function analyzeImage(image) {
  if (!state.modelReady || !state.landmarker) {
    analyzeImageWithFallback(image);
    return;
  }

  setMessage("正在本地识别人脸关键点...", "");
  let result;

  try {
    result = state.landmarker.detect(image);
  } catch (error) {
    console.error(error);
    analyzeImageWithFallback(image);
    return;
  }

  const faces = result.faceLandmarks || [];

  if (!faces.length) {
    $("photoState").textContent = "未识别";
    clearOverlay();
    setMessage("未识别到清晰正脸，请上传光线充足、无遮挡、正面角度的照片。", "warning");
    return;
  }

  const selectedIndex = chooseLargestFace(faces);
  const landmarks = faces[selectedIndex];
  const metrics = calculateMetrics(landmarks);
  metrics.detectionMode = "landmarker";
  metrics.detectionConfidence = "high";
  state.metrics = metrics;

  drawOverlay(landmarks, image);
  renderReport(metrics);

  const multiFaceTip = faces.length > 1 ? "检测到多张人脸，系统已默认选择画面中最大的人脸进行分析。" : "";
  setMessage(`${multiFaceTip} 已在本地完成${landmarks.length}个面部关键点识别，照片未上传服务器。`, "success");
}

function analyzeImageWithFallback(image) {
  setMessage("正在启用本地结构估算模式...", "");
  const detection = createFallbackDetection(image);

  if (!detection) {
    $("photoState").textContent = "未识别";
    clearOverlay();
    setMessage("未识别到清晰正脸，请上传光线充足、无遮挡、正面角度的照片。", "warning");
    return;
  }

  const metrics = calculateMetrics(detection.landmarks);
  metrics.detectionMode = "estimate";
  metrics.detectionConfidence = detection.confidence;
  state.metrics = metrics;

  drawOverlay(detection.landmarks, image);
  renderReport(metrics);

  const confidenceTip = detection.confidence === "low"
    ? "未能稳定定位全部五官，已按画面中心进行结构估算；建议上传光线更均匀的正脸照以获得更稳结果。"
    : "精准关键点模型暂不可用，已使用本地结构估算模式完成分析。";
  setMessage(`${confidenceTip} 照片未上传服务器。`, detection.confidence === "low" ? "warning" : "success");
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;

  try {
    $("photoState").textContent = "读取中";
    setMessage("正在读取照片，照片只会在本地浏览器中处理。", "");
    const { image, dataUrl } = await readImage(file);
    state.image = image;
    refs.photoPreview.src = dataUrl;
    refs.dropZone.classList.add("has-image");
    $("photoState").textContent = "分析中";
    await analyzeImage(image);
  } catch (error) {
    $("photoState").textContent = "读取失败";
    setMessage("照片读取失败，请换一张清晰正脸照片重试。", "warning");
    console.error(error);
  }
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function saveShareCard() {
  if (!state.latestReport) {
    setMessage("请先上传照片完成识别，再保存分享卡片。", "warning");
    return;
  }

  const title = $("faceType").textContent;
  const score = $("scoreNumber").textContent;
  const summary = $("reportSummary").textContent;
  const safeTitle = escapeXml(title);
  const safeScore = escapeXml(score);
  const safeSummary = escapeXml(summary);
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1200" viewBox="0 0 900 1200">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f7f4ec"/>
      <stop offset=".55" stop-color="#edf3f2"/>
      <stop offset="1" stop-color="#dfecea"/>
    </linearGradient>
    <linearGradient id="red" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#822420"/>
      <stop offset=".6" stop-color="#c7504d"/>
      <stop offset="1" stop-color="#b99556"/>
    </linearGradient>
  </defs>
  <rect width="900" height="1200" rx="56" fill="url(#bg)"/>
  <circle cx="160" cy="980" r="230" fill="#c24c62" opacity=".18"/>
  <circle cx="760" cy="170" r="240" fill="#a8c6bf" opacity=".42"/>
  <text x="80" y="130" font-size="34" fill="#172024" font-family="Microsoft YaHei, sans-serif" font-weight="700">观相小馆</text>
  <text x="80" y="240" font-size="64" fill="#172024" font-family="SimSun, serif" font-weight="900">AI全维面相解析</text>
  <circle cx="450" cy="470" r="150" fill="none" stroke="url(#red)" stroke-width="34"/>
  <text x="450" y="494" text-anchor="middle" font-size="88" fill="url(#red)" font-family="Arial, sans-serif" font-weight="900">${safeScore}</text>
  <text x="450" y="555" text-anchor="middle" font-size="24" fill="#738086" font-family="Microsoft YaHei, sans-serif">面相气韵综合分</text>
  <text x="80" y="740" font-size="38" fill="#a93632" font-family="Microsoft YaHei, sans-serif" font-weight="800">${safeTitle}</text>
  <foreignObject x="80" y="790" width="740" height="230">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font:28px/1.7 Microsoft YaHei,sans-serif;color:#3f494d;">${safeSummary}</div>
  </foreignObject>
  <text x="80" y="1085" font-size="20" fill="#738086" font-family="Microsoft YaHei, sans-serif">本卡片仅供传统文化娱乐体验，不构成专业判断。</text>
</svg>`.trim();

  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "guangxiang-share-card.svg";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

refs.photoInput.addEventListener("change", (event) => handleFile(event.target.files[0]));
$("uploadBtn").addEventListener("click", () => refs.photoInput.click());
$("ageInput").addEventListener("change", () => {
  if (state.metrics) renderReport(state.metrics);
});
$("fullReportBtn").addEventListener("click", () => {
  if (!state.latestReport) {
    setMessage("请先上传照片完成识别，再生成完整报告。", "warning");
    return;
  }
  renderFullReport(state.latestReport.full);
  $("fullReportSection").hidden = false;
  $("fullReportSection").scrollIntoView({ behavior: "smooth", block: "start" });
});
$("shareBtn").addEventListener("click", saveShareCard);

["dragenter", "dragover"].forEach((eventName) => {
  refs.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    refs.dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  refs.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    refs.dropZone.classList.remove("dragging");
  });
});

refs.dropZone.addEventListener("drop", (event) => {
  handleFile(event.dataTransfer.files[0]);
});

window.addEventListener("resize", () => {
  if (state.image && state.metrics && refs.photoPreview.complete) {
    analyzeImage(state.image);
  }
});

initFaceLandmarker();
