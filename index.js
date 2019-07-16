const cv = require('opencv4nodejs')
const path = require('path')
const fs = require('fs')
const sharp = require('sharp')

const imgsPath = path.resolve('./images')
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2)

fs.readdirSync(imgsPath)
  .map((file) => path.resolve(imgsPath, file))
  .forEach((filePath) => {
    if (!/(jpg|jpeg|png)$/i.test(filePath)) return

    const img = cv.imread(filePath) // read image
    const grayImg = img.bgrToGray() // face recognizer works with gray scale images
    const faceRects = classifier.detectMultiScale(grayImg).objects

    if (!faceRects.length) throw new Error('failed to detect faces')

    faceRects.forEach(({ height, width, x, y }, i) => {
      const padding = Math.floor(height * 0.3)
      const left = x - padding
      const top = y - padding
      sharp(filePath)
        .extract({
          width: width + padding,
          height: height + padding,
          left: left < 0 ? x : left,
          top: top < 0 ? y : top
        })
        .resize({ width: 200 })

        // save thumbnails to faces folder
        //   .toFile(`./faces/${i}_${x}_${y}.jpg`)
        //   .then(info => {
        //     console.log('saved', info)
        //   })

        // save to html as base64 data image
        .toBuffer()
        .then((data) => {
          fs.appendFileSync(
            'faces.html',
            `<img src="data:image/png;base64,${data.toString('base64')}" />`
          )
        })

        // catch error
        .catch((error) => {
          console.log('An error occurred', error)
        })
    })
  })
