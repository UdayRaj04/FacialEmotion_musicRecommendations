
export async function predictEmotion(file) {
  if (!file) throw new Error('No image file provided')

  const form = new FormData()
  form.append('image', file)

  const url = `${import.meta.env.VITE_API_BASE || 'http://localhost:5000'}/predict`
  const res = await fetch(url, { method: 'POST', body: form })
  const data = await res.json()

  if (!res.ok) {
    // Show alert if no face
    if (data?.error === "Face not in frame") {
      alert("Face not detected! Please keep your face in frame.")
    }
    throw new Error(data?.error || 'Prediction failed')
  }

  return data
}

// export async function predictEmotion(file) {
//   if (!file) throw new Error('No image file provided')
//   const form = new FormData()
//   form.append('image', file)

//   const url = `${import.meta.env.VITE_API_BASE || 'http://localhost:5000'}/predict`
//   const res = await fetch(url, { method: 'POST', body: form })
//   const data = await res.json()
//   if (!res.ok) {
//     throw new Error(data?.error || 'Prediction failed')
//   }
//   return data
// }
