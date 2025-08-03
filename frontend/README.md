# Tetra WebGUI Frontend

A modern Vue.js frontend for monitoring database health and querying data using natural language.

## Features

- **Database Monitor**: Real-time visualization of data coverage across all schemas
- **Chat Interface**: Natural language to SQL conversion powered by LLM
- **Interactive Charts**: Automatic data visualization with Plotly.js
- **SQL Syntax Highlighting**: Beautiful code display with Prism.js
- **Real-time Updates**: WebSocket connection for live data

## Development Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Variables

Create a `.env.development` file for local development:

```env
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/ws/monitor
```

## Project Structure

```
src/
├── components/     # Reusable Vue components
├── views/          # Page components
├── services/       # API and WebSocket services
├── router/         # Vue Router configuration
└── main.js        # Application entry point
```

## Tech Stack

- **Vue 3**: Progressive JavaScript framework
- **Vite**: Fast build tool
- **Tailwind CSS**: Utility-first CSS framework
- **Plotly.js**: Interactive charting library
- **Prism.js**: Syntax highlighting
- **Axios**: HTTP client

## Backend Requirements

This frontend requires the FastAPI backend to be running on port 8000. See the backend README for setup instructions.
