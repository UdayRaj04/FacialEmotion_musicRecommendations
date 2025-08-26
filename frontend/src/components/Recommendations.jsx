
import React, { useEffect, useState } from 'react'

export default function Recommendations({ emotion, source }) {
  const [items, setItems] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    async function run() {
      setError(null)
      try {
        const url = `${import.meta.env.VITE_API_BASE || 'http://localhost:5000'}/recommend?emotion=${encodeURIComponent(emotion)}&source=${source}`
        const res = await fetch(url)
        const data = await res.json()
        if (data.items) setItems(data.items)
        else setItems([])
      } catch (e) {
        setError('Could not load recommendations.')
      }
    }
    if (emotion) run()
  }, [emotion, source])

  return (
    <div className="recs">
      <h3>Recommendations</h3>
      {error && <div className="error">{error}</div>}
      <div className="grid">
        {items.map((it, idx) => (
          <div key={idx} className="embed">
            {source === 'youtube' ? (
              <iframe
                width="100%"
                height="220"
                src={`https://www.youtube.com/embed/${it.youtubeId}`}
                title={it.title}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
              ></iframe>
            ) : (
              <iframe
                style={{borderRadius: '12px'}}
                src={`https://open.spotify.com/embed/${it.spotifyUri.replace('spotify:', '').replace(':', '/')}`}
                width="100%"
                height="232"
                frameBorder="0"
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"
              ></iframe>
            )}
            <div className="caption">{it.title}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
