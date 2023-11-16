import * as ort from 'onnxruntime-web'

async function main() {
  const session = await ort.InferenceSession.create("/model.onnx");

  const webcam = document.getElementById("webcam");
  const display = document.getElementById("display");

  // Virtual hidden canvas to draw images from the video stream so we can
  // capture individual frames as ImageData objects.
  const canvas = document.createElement("canvas");
  const canvasCtx = canvas.getContext("2d", { willReadFrequently: true });
  canvas.style.display = 'none';

  async function processFrame() {
    // Get current webcam frame.
    canvasCtx.drawImage(webcam, 0, 0, 224, 224);
    const frame = canvasCtx.getImageData(0, 0, 224, 224);

    // Convert to Float32 tensor.
    const inputTensor = await ort.Tensor.fromImage(frame);

    // Run the model.
    const outputTensor = (await session.run({ "input1": inputTensor })).output1;

    // Post-processing output tensor (clamp values between 0, 255 and divide by 255).
    for (let i = 0; i < outputTensor.size; i++) {
      outputTensor.data[i] = Math.min(Math.max(0, outputTensor.data[i]), 255) / 255.0;
    }

    // Convert Float32 image on range [0, 1] back to ImageData.
    const outImage = outputTensor.toImageData();

    // Display image.
    display.getContext("2d").putImageData(outImage, 0, 0);

    // Process the next frame.
    window.requestAnimationFrame(processFrame);
  }

  // Get webcam stream and send it to the webcam video element.
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then((stream) => {
      webcam.srcObject = stream;
      webcam.play();
    })

  // Once the first frame is loaded to the webcam video element, set the
  // width,height of all the other elements and start processing the frames.
  webcam.addEventListener("loadeddata", (_) => {
    document.getElementById("description").innerHTML = `A simple demo of ONNX Runtime for Web running a neural style transfer model on webcam inputs (<a href="https://github.com/jaymody/onnxruntime-web-example">source code</a>)`
    webcam.setAttribute("width", webcam.videoWidth);
    webcam.setAttribute("height", webcam.videoHeight);
    canvas.setAttribute("width", webcam.videoWidth);
    canvas.setAttribute("height", webcam.videoHeight);
    display.setAttribute("width", 224);
    display.setAttribute("height", 224);
    processFrame();
  });
}

(async () => {
  try {
    await main()
  } catch (err) {
    console.error(err);
    document.getElementById("description").innerText = "An error was encountered."
  }
})()
