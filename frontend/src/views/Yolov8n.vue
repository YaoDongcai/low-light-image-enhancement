<template>
    <div class="container">
        <Loader v-if="loading">
            {{ loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text }}
        </Loader>
        <div class="header">
            <h1>YOLOv8 Object Detection App</h1>
            <p>
                YOLOv8 object detection application live on browser powered by
                <code>onnxruntime-web</code>
            </p>
            <p>
                Serving : <code class="code">{{ modelName }}</code>
            </p>
        </div>

        <div class="content">
            <img ref="imageRef" src="#" alt="" :style="{ display: image ? 'block' : 'none' }" @load="onImageLoad" />
            <canvas id="canvas" :width="modelInputShape[2]" :height="modelInputShape[3]" ref="canvasRef"></canvas>
        </div>

        <input type="file" ref="inputImage" accept="image/*" style="display: none" @change="onImageChange" />
        <div class="btn-container">
            <button @click="openImage">Open local image</button>
            <button v-if="image" @click="closeImage">Close image</button>
        </div>
    </div>
</template>
<style lang="scss">
.header {
    text-align: center;
}

.header p {
    margin: 5px 0;
}

.code {
    padding: 5px;
    color: greenyellow;
    background-color: black;
    border-radius: 5px;

}

.content {
    position: relative;

    img {
        width: 100%;
        max-width: 720px;
        max-height: 500px;
        border-radius: 10px;
    }

    canvas {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }
}

button {
    text-decoration: none;
    color: white;
    background-color: black;
    border: 2px solid black;
    margin: 0 5px;
    padding: 5px;
    border-radius: 5px;
    cursor: pointer;
}

button:hover {
    color: black;
    background-color: white;
    border: 2px solid black;
}
</style>

<script  setup>

import { ref, reactive, onMounted } from 'vue';
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession, env} from "onnxruntime-web";
import Loader from "@/components/Loader/index.vue";
import { detectImage } from "@/utils/detect";
import { download } from "@/utils/download";

// Refs
const session = ref(null);
const loading = ref({ text: "Loading OpenCV.js", progress: null });
const image = ref(null);
const inputImage = ref(null);
const imageRef = ref(null);
const canvasRef = ref(null);

// Configs
const modelName = "yolov8n.onnx";
const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.25;
env.wasm.wasmPaths = {
  'ort-wasm.wasm': './ort-wasm.wasm',
  'ort-wasm-simd.wasm': './ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm': './ort-wasm-threaded.wasm'
}
// Methods
const initializeModel = async () => {
    const baseModelURL = `/model`;

    // create session
    const arrBufNet = await download(
        `${modelName}`,
        [(text, progress) => loading.value = { text, progress }]
    );
    console.log('----arrBufNet----', arrBufNet);
    const yolov8 = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
        `nms-yolov8.onnx`,
        [(text, progress) => loading.value = { text, progress }]
    );
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup main model
    loading.value = { text: "Warming up model...", progress: null };
    const tensor = new Tensor(
        "float32",
        new Float32Array(modelInputShape.reduce((a, b) => a * b)),
        modelInputShape
    );
    await yolov8.run({ images: tensor });

    session.value = { net: yolov8, nms: nms };
    loading.value = null;
};

const onImageLoad = () => {
    detectImage(
        imageRef.value,
        canvasRef.value,
        session.value,
        topk,
        iouThreshold,
        scoreThreshold,
        modelInputShape
    );
};

const onImageChange = (e) => {
    if (image.value) {
        URL.revokeObjectURL(image.value);
        image.value = null;
    }

    const url = URL.createObjectURL(e.target.files[0]);
    imageRef.value.src = url;
    image.value = url;
};

const openImage = () => {
    inputImage.value.click();
};

const closeImage = () => {
    inputImage.value.value = "";
    imageRef.value.src = "#";
    URL.revokeObjectURL(image.value);
    image.value = null;
};

// Lifecycle
onMounted(() => {
    cv["onRuntimeInitialized"] = initializeModel;
});
</script>