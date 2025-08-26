import React, { useEffect, useRef, useState } from "react"

export default function CameraCapture({ onCapture }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [supported, setSupported] = useState(true)
  const [isClicked, setIsClicked] = useState(false)

  useEffect(() => {
    let stream

    async function init() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
        setSupported(true)
      } catch (err) {
        console.error("Camera error:", err)
        setSupported(false)
      }
    }

    init()

    // Cleanup on unmount
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const snap = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

     // Animate button
    setIsClicked(true)
    setTimeout(() => setIsClicked(false), 200) // remove class after animation


    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext("2d")
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    canvas.toBlob(blob => {
      if (blob && onCapture) {
        onCapture(new File([blob], "capture.jpg", { type: "image/jpeg" }))
      }
    }, "image/jpeg", 0.9)
  }

  return (
    <div className="card">
      <h3>Webcam</h3>
      {supported ? (
        <>
          <video ref={videoRef} autoPlay playsInline className="video" />
          {/* <button onClick={snap}>📸 Capture</button> */}
          <button
            onClick={snap}
            className={`capture-btn ${isClicked ? "clicked" : ""}`}
          >
            📸 Capture
          </button>
          <canvas ref={canvasRef} style={{ display: "none" }} />
        </>
      ) : (
        <p>❌ Camera not available or permission denied.</p>
      )}
    </div>
  )
}
