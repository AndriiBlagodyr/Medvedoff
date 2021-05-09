import './App.css';
import React from 'react';
import * as tf from '@tensorflow/tfjs';
tf.setBackend('webgl');

const pascalvoc = [
	[0, 0, 0],
	[128, 0, 0],
	[0, 128, 0]
];

async function load_model() {
	const model = await tf.loadLayersModel('http://127.0.0.1:8080/model.json');
	return model;
}

const modelPromise = load_model();

class App extends React.Component {
	videoRef = React.createRef();
	canvasRef = React.createRef();

	componentDidMount() {
		// eslint-disable-next-line
		console.log(navigator);
		if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
			const webCamPromise = navigator.mediaDevices
				.getUserMedia({
					audio: false,
					video: {
						facingMode: 'user'
					}
				})
				.then(stream => {
					window.stream = stream;
					this.videoRef.current.srcObject = stream;
					// eslint-disable-next-line
					return new Promise((resolve, reject) => {
						this.videoRef.current.onloadedmetadata = () => {
							resolve();
						};
					});
				});
			Promise.all([modelPromise, webCamPromise])
				.then(values => {
					this.detectFrame(this.videoRef.current, values[0]);
				})
				.catch(error => {
					// eslint-disable-next-line
					console.error(error);
				});
		}
	}

	detectFrame = (video, model) => {
		tf.engine().startScope();
		const predictions = model.predict(this.process_input(video));
		this.renderPredictions(predictions);
		requestAnimationFrame(() => {
			this.detectFrame(video, model);
		});
		tf.engine().endScope();
	};

	process_input(video_frame) {
		const img = tf.browser
			.fromPixels(video_frame)
			.toFloat()
			.resizeBilinear([320, 320]);
		const batched = img.expandDims();
		// eslint-disable-next-line
		console.log(batched);
		return batched;
	}

	renderPredictions = async predictions => {
		const img_shape = [400, 400]; // before 320 * 320
		const offset = 0;
		const segmPred = tf.image.resizeBilinear(predictions, img_shape);
		const segmMask = segmPred.argMax(3).reshape(img_shape);
		const width = segmMask.shape.slice(0, 1);
		const height = segmMask.shape.slice(1, 2);
		const data = await segmMask.data();
		const bytes = new Uint8ClampedArray(width * height * 4);
		for (let i = 0; i < height * width; ++i) {
			const partId = data[i];
			const j = i * 4;
			if (partId === -1) {
				bytes[j + 0] = 255;
				bytes[j + 1] = 255;
				bytes[j + 2] = 255;
				bytes[j + 3] = 255;
			} else {
				const color = pascalvoc[partId + offset];

				if (!color) {
					throw new Error(`No color could be found for part id ${partId}`);
				}
				bytes[j + 0] = color[0];
				bytes[j + 1] = color[1];
				bytes[j + 2] = color[2];
				bytes[j + 3] = 255;
			}
		}
		const out = new ImageData(bytes, width, height);
		const ctx = this.canvasRef.current.getContext('2d');
		ctx.scale(1.5, 1.5);
		ctx.putImageData(out, 520, 60);
	};

	render() {
		return (
			<div>
				<h1 style={{ textAlign: 'center', fontSize: '48px' }}>Raspberry Live Detector</h1>
				<div style={{ display: 'flex', justifyContent: 'center' }}>
					<video
						autoPlay
						playsInline
						muted
						ref={this.videoRef}
						width="480"
						height="480"
						style={{ borderColor: 'transparent', marginLeft: '100px' }}
					/>
					<canvas
						ref={this.canvasRef}
						width="960"
						height="480"
						style={{ marginLeft: '-380px', position: 'relative', top: '-25px' }}
					/>
				</div>
			</div>
		);
	}
}

export default App;
