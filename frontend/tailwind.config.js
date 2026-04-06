/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['Outfit', 'sans-serif'],
        body:    ['DM Sans', 'sans-serif'],
        mono:    ['Fira Code', 'monospace'],
      },
      animation: {
        'fade-in':     'fade-page 0.35s ease forwards',
        'slide-up':    'slide-up 0.25s ease forwards',
        'spin-slow':   'spin 2s linear infinite',
        'glow-pulse':  'glow-pulse 3s ease-in-out infinite',
        'orb-float':   'orb-float 20s ease-in-out infinite',
        'shimmer':     'shimmer 1.6s infinite linear',
      },
    },
  },
  plugins: [],
}
