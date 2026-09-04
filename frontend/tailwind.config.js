/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        base: '#F8FAFC',
        panel: '#FFFFFF',
        card: '#FFFFFF',
        elevated: '#F5F7FA',
        brand: {
          50: '#EFF6FF',
          100: '#DBEAFE',
          500: '#3B82F6',
          primary: '#155E9E',
          secondary: '#0F8B8D',
        },
        risk: {
          low: '#22C55E',
          medium: '#F59E0B',
          high: '#EF4444',
          extreme: '#991B1B',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
