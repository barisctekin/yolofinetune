import express from 'express';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';

dotenv.config();

const imageDir = path.join(path.resolve(), 'images');
const outputDir = path.join(path.resolve(), 'output');

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
}

const app = express();

app.post('/mask-plates', async (req, res) => {
    try {
        const files = fs.readdirSync(imageDir);

        for (const file of files) {
            const filePath = path.join(imageDir, file);
            const outputFilePath = path.join(outputDir, file);

            await new Promise((resolve, reject) => {
                execFile('python', ['yolo_license_plate.py', filePath, outputFilePath, '--debug'], (error, stdout, stderr) => {
                    if (error) {
                        console.error(`Error: ${stderr}`);
                        reject(error);
                    } else {
                        console.log(`Processed ${file}`);
                        console.log(stdout);
                        resolve(stdout);
                    }
                });
            });
        }

        res.status(200).send('License plates masked where necessary');
    } catch (error) {
        console.error('An error occurred:', error);
        res.status(500).send('An error occurred');
    }
});

const port = 3000;
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
