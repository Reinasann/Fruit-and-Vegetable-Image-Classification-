const express = require('express');
const line = require('@line/bot-sdk');
const tf = require('@tensorflow/tfjs-node');

const config = {
    channelAccessToken: process.env.CHANNEL_ACCESS_TOKEN,
    channelSecret: process.env.CHANNEL_SECRET,
};
const modelUrl = 'https://teachablemachine.withgoogle.com/models/2kVdl0QYs/model.json';
const classNames = ['มะเขือเทศปกติ', 'มะเขือเทศเน่า', 'สตรอเบอร์รี่ปกติ', 'สตรอเบอร์รี่เน่า', 'มันฝรั่งปกติ', 'มันฝรั่งเน่า', 'ทับทิมปกติ', 'ทับทิมเน่า', 'ส้มปกติ', 'ส้มเน่า', 'มะม่วงปกติ', 'มะม่วงเน่า', 'พุทราจีนปกติ', 'พุทราจีนเน่า', 'ฝรั่งปกติ', 'ฝรั่งเน่า', 'องุ่นปกติ', 'องุ่นเน่า', 'แตงกวาปกติ', 'แตงกวาเน่า', 'แครอทปกติ', 'แครอทเน่า', 'พริกหวานปกติ', 'พริกหวานเน่า', 'กล้วยปกติ', 'กล้วยเน่า', 'แอปเปิ้ลปกติ', 'แอปเปิ้ลเน่า'];

const app = express();
const client = new line.Client(config);
let model;

app.post('/webhook', line.middleware(config), (req, res) => {
    Promise.all(req.body.events.map(handleEvent))
        .then((result) => res.json(result))
        .catch((err) => {
            console.error(err);
            res.status(500).end();
        });
});

async function handleEvent(event) {
    if (event.type !== 'message' || event.message.type !== 'image') {
        return Promise.resolve(null);
    }
    try {
        const imageBuffer = await getImageBufferFromLine(event.message.id);

        // ===============================================================
        // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        //                  >>> จุดที่แก้ไข <<<
        //       เพิ่ม .div(tf.scalar(127.5)).sub(tf.scalar(1))
        //   เพื่อปรับค่าสีของรูปภาพให้เหมือนกับตอนที่เทรนโมเดล
        // ===============================================================

        const imageTensor = tf.node.decodeImage(imageBuffer, 3)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(127.5)) // <-- เพิ่มบรรทัดนี้
            .sub(tf.scalar(1))     // <-- และบรรทัดนี้
            .expandDims();

        // ===============================================================
        // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        // ===============================================================

        const predictionResult = await model.predict(imageTensor).data();

        let bestPrediction = { className: 'ไม่รู้จัก', probability: 0 };
        for (let i = 0; i < predictionResult.length; i++) {
            if (predictionResult[i] > bestPrediction.probability) {
                bestPrediction.probability = predictionResult[i];
                bestPrediction.className = classNames[i];
            }
        }

        const confidence = Math.round(bestPrediction.probability * 100);
        const replyFlex = {
            type: 'flex',
            altText: 'ผลการตรวจสอบภาพ',
            contents: {
                type: 'bubble',
                size: 'kilo',
                hero: {
                    type: 'image',
                    url: 'https://cdn-icons-png.flaticon.com/512/415/415682.png', // ไอคอนรูปผลไม้ (ปรับได้)
                    size: 'full',
                    aspectRatio: '1:1',
                    aspectMode: 'cover'
                },
                body: {
                    type: 'box',
                    layout: 'vertical',
                    contents: [
                        {
                            type: 'text',
                            text: `🔍 ผลการตรวจสอบ`,
                            weight: 'bold',
                            size: 'lg',
                            align: 'center',
                            color: '#E67E22'
                        },
                        {
                            type: 'text',
                            text: `ฉันคิดว่ารูปนี้คือ...`,
                            size: 'sm',
                            align: 'center',
                            margin: 'md'
                        },
                        {
                            type: 'text',
                            text: `"${bestPrediction.className}"`,
                            weight: 'bold',
                            size: 'xl',
                            color: '#2ECC71',
                            align: 'center',
                            wrap: true,
                            margin: 'md'
                        },
                        {
                            type: 'text',
                            text: `ความมั่นใจ: ${confidence}%`,
                            size: 'sm',
                            align: 'center',
                            color: '#555555',
                            margin: 'sm'
                        }
                    ]
                }
            }
        };


        return client.replyMessage(event.replyToken, replyFlex);

    } catch (error) {
        console.error(error);
        return client.replyMessage(event.replyToken, { type: 'text', text: 'ขออภัยค่ะ เกิดข้อผิดพลาดบางอย่าง' });
    }
}

function getImageBufferFromLine(messageId) {
    return new Promise((resolve, reject) => {
        client.getMessageContent(messageId)
            .then((stream) => {
                const chunks = [];
                stream.on('data', (chunk) => { chunks.push(chunk); });
                stream.on('error', (err) => { reject(err); });
                stream.on('end', () => { resolve(Buffer.concat(chunks)); });
            });
    });
}

async function startServer() {
    try {
        console.log('Loading model...');
        model = await tf.loadLayersModel(modelUrl);
        console.log('Model loaded!');
        const port = process.env.PORT || 3000;
        app.listen(port, () => {
            console.log(`Bot is ready on port ${port}`);
        });
    } catch (error) {
        console.error('Failed to load model:', error);
    }
}

startServer();