"use client"

import { useEffect, useRef } from "react"
import * as THREE from "three"

interface WaveAnimationProps {
  width?: number
  height?: number
  particles?: number
  pointSize?: number
  waveSpeed?: number
  baseWaveIntensity?: number 
  particleColor?: string
  gridDistance?: number
  className?: string
  analyserNode?: AnalyserNode | null
}

export function WaveAnimation({
  width,
  height,
  particles = 5000,
  pointSize = 1.5,
  waveSpeed = 2.0,
  baseWaveIntensity = 8.0, 
  particleColor = "#ffffff",
  gridDistance = 5,
  className = "",
  analyserNode = null,
}: WaveAnimationProps) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const animationIdRef = useRef<number | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return

    const container = canvasRef.current
    const w = width || container.clientWidth || window.innerWidth
    const h = height || container.clientHeight || window.innerHeight
    const dpr = window.devicePixelRatio

    const fov = 60
    const fovRad = (fov / 2) * (Math.PI / 180)
    const dist = h / 2 / Math.tan(fovRad)

    const startTime = performance.now()

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(w, h)
    renderer.setClearColor(0x000000, 0) // transparent background
    renderer.setPixelRatio(dpr)
    rendererRef.current = renderer

    container.appendChild(renderer.domElement)

    const camera = new THREE.PerspectiveCamera(fov, w / h, 1, dist * 2)
    camera.position.set(0, 0, 10)

    const scene = new THREE.Scene()

    const geo = new THREE.BufferGeometry()
    const positions: number[] = []

    const gridWidth = 400 * (w / h)
    const depth = 400

    for (let x = 0; x < gridWidth; x += gridDistance) {
      for (let z = 0; z < depth; z += gridDistance) {
        positions.push(-gridWidth / 2 + x, -30, -depth / 2 + z)
      }
    }

    const positionAttribute = new THREE.Float32BufferAttribute(positions, 3)
    geo.setAttribute("position", positionAttribute)

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        u_time: { value: 0.0 },
        u_point_size: { value: pointSize },
        u_color: { value: new THREE.Color(particleColor) },
        u_intensity: { value: baseWaveIntensity },
        u_speed: { value: waveSpeed }
      },
      vertexShader: `
        #define M_PI 3.1415926535897932384626433832795
        precision mediump float;
        uniform float u_time;
        uniform float u_point_size;
        uniform float u_intensity;
        uniform float u_speed;
        
        void main() {
          vec3 p = position;
          p.y += (
            cos(p.x / M_PI * u_intensity + u_time * u_speed) +
            sin(p.z / M_PI * u_intensity + u_time * u_speed)
          );
          gl_PointSize = u_point_size;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: `
        precision mediump float;
        uniform vec3 u_color;
        
        void main() {
          gl_FragColor = vec4(u_color, 1.0);
        }
      `,
    })

    const mesh = new THREE.Points(geo, mat)
    scene.add(mesh)

    const dataArray = new Uint8Array(128)

    function render() {
      const time = (performance.now() - startTime) * 0.001
      mesh.material.uniforms.u_time.value = time
      
      // Dynamic modulation
      let currentIntensity = baseWaveIntensity
      if (analyserNode) {
        analyserNode.getByteFrequencyData(dataArray)
        let sum = 0
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i]
        const volRatio = (sum / dataArray.length) / 255.0
        currentIntensity = baseWaveIntensity + volRatio * 60.0
      }
      // Smooth intensity
      mesh.material.uniforms.u_intensity.value +=
        (currentIntensity - mesh.material.uniforms.u_intensity.value) * 0.12
      // Smooth colour transition
      const targetColor = new THREE.Color(particleColor)
      mesh.material.uniforms.u_color.value.lerp(targetColor, 0.05)

      renderer.render(scene, camera)
      animationIdRef.current = requestAnimationFrame(render)
    }

    render()

    const handleResize = () => {
      const newW = width || container.clientWidth || window.innerWidth
      const newH = height || container.clientHeight || window.innerHeight
      camera.aspect = newW / newH
      camera.updateProjectionMatrix()
      renderer.setSize(newW, newH)
    }

    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
      if (rendererRef.current) {
        container.removeChild(rendererRef.current.domElement)
        rendererRef.current.dispose()
      }
      geo.dispose()
      mat.dispose()
    }
  }, [width, height, particles, pointSize, waveSpeed, baseWaveIntensity, particleColor, gridDistance, analyserNode])

  return (
    <div
      ref={canvasRef}
      className={className}
      style={{
        width: width ? `${width}px` : "100%",
        height: height ? `${height}px` : "100%",
        overflow: "hidden",
      }}
    />
  )
}
