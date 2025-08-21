# NuRea IDE

A powerful Electron-based IDE specifically designed for the NuRea simulation platform. Features Monaco editor, integrated terminal, AI chat, and Docker Compose management.

## Features

- **Monaco Editor**: Full-featured code editor with syntax highlighting for Python, Julia, JSON, and more
- **Integrated Terminal**: xterm.js terminal for running commands and viewing output
- **AI Chat**: OpenAI-compatible AI assistant for code explanation, refactoring, and testing
- **File Explorer**: Browse and edit files in your workspace
- **Docker Integration**: Start/stop Julia backend services directly from the IDE
- **Search**: Text and regex search across your codebase
- **Diff View**: Compare and apply changes with built-in diff viewer

## Prerequisites

- Node.js 18+ 
- Docker Desktop
- Python 3.11+
- (Optional) OPENAI_API_KEY environment variable for AI features

## Installation

1. Navigate to the IDE directory:
   ```bash
   cd ide
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the IDE:
   ```bash
   npm start
   ```

## Usage

### Basic Navigation
- **Sidebar**: Browse files, manage Docker services, and search code
- **Editor**: Main coding area with Monaco editor
- **Terminal**: Integrated terminal for running commands
- **AI Chat**: Right panel for AI assistance

### Docker Management
- **Start Julia Backend**: Launches the Julia optimization service
- **Stop Julia Backend**: Stops the service
- **Health Check**: Shows backend status

### AI Features
- **AI Explain**: Get explanations of selected code
- **AI Refactor**: AI-powered code refactoring suggestions
- **AI Tests**: Generate unit tests for your code
- **Chat**: General AI assistance

### File Operations
- **Save**: Save current file (Ctrl+S)
- **Diff Toggle**: Switch between editor and diff view
- **Search**: Search across your codebase with text or regex

## Building

### Package for Distribution
```bash
npm run pack
```

### Create Installer
```bash
npm run make
```

## Configuration

Edit `config.json` to customize:
- Workspace root path
- Docker Compose file location
- Orchestrator script and plan paths
- AI provider settings
- Julia backend health check URL

## Architecture

- **Main Process** (`main.js`): Handles IPC, file operations, Docker commands
- **Renderer Process** (`renderer/`): UI components and Monaco editor
- **Preload Script** (`preload.js`): Secure bridge between processes
- **IPC**: Secure communication between main and renderer processes

## Security

- Context isolation enabled
- Node integration disabled
- Sandbox mode enabled
- Secure preload script for IPC

## Troubleshooting

### Common Issues

1. **Monaco Editor not loading**: Check that all Monaco dependencies are installed
2. **Terminal not working**: Ensure xterm.js is properly loaded
3. **Docker commands failing**: Verify Docker is running and accessible
4. **AI chat errors**: Check OPENAI_API_KEY environment variable

### Debug Mode

Run with debug logging:
```bash
DEBUG=* npm start
```

## Contributing

The IDE is built with:
- **Electron**: Cross-platform desktop app framework
- **Monaco Editor**: VS Code's editor component
- **xterm.js**: Terminal emulator
- **Vanilla JavaScript**: No framework dependencies

## License

Apache 2.0 - Same as the main NuRea project
