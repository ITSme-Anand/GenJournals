import { useState, useEffect } from 'react'



function App() {
  const [count, setCount] = useState(0)

  useEffect(() => {
  fetch("http://localhost:8000/health")
    .then(res => res.json())
    .then(()=>console.log("backend running successfully"))
    .catch((err)=>{console.log(err)})
  }, [])

  return (
    <div> Hi There!</div>
  )
}

export default App
