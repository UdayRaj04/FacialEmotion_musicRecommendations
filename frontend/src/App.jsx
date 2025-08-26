
import React, { useState ,useEffect } from 'react'
import CameraCapture from './components/CameraCapture.jsx'
import Recommendations from './components/Recommendations.jsx'
import { predictEmotion } from './api.js'
import Footer from './components/Footer.jsx'

export default function App() {
  const [imageFile, setImageFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const [loading, setLoading] = useState(false)
  const [hintKey, setHintKey] = useState(0) // used to trigger refresh animation

  // const handlePredict = async () => {
  //   setError(null)
  //   try {
  //     const res = await predictEmotion(imageFile)
  //     setResult(res)
  //   } catch (e) {
  //     setError(e.message || 'Prediction failed')
  //     setResult(null)
  //   }
  // }

  const handlePredict = async () => {
  if (!imageFile) return
  setError(null)
  setLoading(true)
  try {
    const res = await predictEmotion(imageFile)

    if (res?.error === "Face not in frame") {
      alert("⚠️ Face not detected! Please keep your face in frame.")  // <-- Show alert
      setResult(null)
      return
    }

    setResult(res)
  } catch (e) {
    setError(e.message || 'Prediction failed')
    setResult(null)
  } finally {
    setLoading(false)
  }
}


  // const handlePredict = async () => {
  //   if (!imageFile) return
  //   setError(null)
  //   setLoading(true)
  //   try {
  //     const res = await predictEmotion(imageFile)
  //     setResult(res)
  //   } catch (e) {
  //     setError(e.message || 'Prediction failed')
  //     setResult(null)
  //   } finally {
  //     setLoading(false)
  //   }
  // }

  // Trigger refresh animation whenever imageFile changes
  useEffect(() => {
    if (imageFile) {
      setHintKey(prev => prev + 1)
    }
  }, [imageFile])

  return (
    <>
    <div className="container">
    {/* <video className="video-bg" autoPlay muted loop>
    <source src="/public/video.mp4" type="video/mp4" />
    Your browser does not support the video tag.
  </video> */}
      <header>
      <div className='logo-con'>
      <img src="logo.png" alt="URS" className="logo-img" />
                <h1 className='title'> EmotiVibe</h1>
                </div>
        <p>Webcam facial emotion → music recommendations</p>
      </header>

      <section className="input-grid">
        <CameraCapture onCapture={setImageFile} />
        <div className="card">
          <h3>Instructions</h3>
          <ol>
            <li>Allow camera permission and click <b>Capture</b></li>
            <li>Click <b>Detect Emotion</b></li>
            <li>Switch between YouTube and Spotify</li>
          </ol>
        </div>
      </section>

      {/* <div className="actions">
        <button disabled={!imageFile} onClick={handlePredict}>Detect Emotion</button>
        {imageFile && <span className="hint">Image ready ✓</span>}
      </div> */}
      <div className="actions">
        <button disabled={!imageFile || loading} onClick={handlePredict}>
          {loading ? (
            <span className="loader"></span>  // spinner
          ) : (
            "Detect Emotion"
          )}
        </button>

        {imageFile && (
          <span key={hintKey} className="hint animate-refresh">
            Image ready ✓
          </span>
        )}
      </div>

      {error && <div className="error">{error}</div>}
      {result && <Results result={result} />}
    </div>
    <Footer/></>
  )
}

function Results({ result }) {
  const [source, setSource] = useState('youtube')
  const [key, setKey] = useState(0)

  const emotion = result?.emotion
  const scores = result?.scores || {}

  return (
    <section className="results">
      <div className="card">
        <h2>Detected Emotion: <span className="pill">{emotion}</span></h2>
        <div className="probs">
          {Object.entries(scores).sort((a,b)=>b[1]-a[1]).map(([k,v])=> (
            <div key={k} className="probrow">
              <span>{k}</span>
              <progress max="1" value={v}></progress>
              <span>{(v*100).toFixed(1)}%</span>
            </div>
          ))}
        </div>

        <div className="switcher">
          <label>Source:</label>
          <select value={source} onChange={e=>{setSource(e.target.value); setKey(x=>x+1)}}>
            <option value="youtube">YouTube</option>
            <option value="spotify">Spotify</option>
          </select>
        </div>
      </div>

      <Recommendations key={key} emotion={emotion} source={source} />
    </section>
  )
}
