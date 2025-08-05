# Open Finance Decision Support API

A Fastify-based API that implements various Multi-Criteria Decision Analysis (MCDA) methods to support financial decision-making processes.

## Features

- **FHP (Fuzzy Hierarchical Process)**: Implemented and ready to use
- **Optimization**: Multi-objective optimization algorithms for financial decisions
- **SVI (Strategic Value Index)**: Strategic value assessment and indexing

## Prerequisites

- Node.js 18 or higher
- npm (Node package manager)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd openstrat
```

2. Install the required dependencies:
```bash
npm install
```

## Running the Application

The application can be run in several ways:

### Development Mode
```bash
# Build TypeScript and watch for changes
npm run dev:build

# Run with nodemon for auto-restart
npm run dev

# Or run both together
npm run dev:full
```

### Production Mode
```bash
# Build the application
npm run build

# Start the server
npm start
```

### Environment Variables

You can configure the application using environment variables:

```bash
export PORT=3000              # Server port (default: 3000)
export HOST=0.0.0.0          # Server host (default: 0.0.0.0)
export NODE_ENV=production   # Environment mode
```

## API Documentation

Once the server is running, you can access the API documentation at:

- **Scalar API Reference**: `http://localhost:3000/reference` (recommended)
- **Swagger UI**: `http://localhost:3000/documentation`
- **OpenAPI JSON**: `http://localhost:3000/openapi.json`

### Available Endpoints

#### FHP (Fuzzy Hierarchical Process)
- **Endpoint**: `/fhp`
- **Methods**: GET, POST
- **Description**: Fuzzy Hierarchical Process for multi-criteria decision analysis
- **GET**: Retrieve FHP data and status
- **POST**: Create new FHP analysis with criteria and values

#### Optimization
- **Endpoint**: `/optimization`
- **Methods**: GET, POST
- **Description**: Multi-objective optimization algorithms for financial decision support
- **GET**: Retrieve optimization data and status
- **POST**: Execute optimization algorithms with parameters

#### SVI (Strategic Value Index)
- **Endpoint**: `/svi`
- **Methods**: GET, POST
- **Description**: Strategic Value Index for assessing investment opportunities
- **GET**: Retrieve SVI data and status
- **POST**: Create new SVI entries with index values and dates

## Development

### Project Structure
```
openstrat/
├── src/
│   └── app.ts              # Main Fastify application
├── dist/                   # Compiled JavaScript (generated)
├── package.json           # Project dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── .gitignore           # Git ignore rules
└── README.md           # This file
```

### Available Scripts

- `npm run build` - Compile TypeScript to JavaScript
- `npm start` - Run the compiled application
- `npm run dev` - Run with nodemon for development
- `npm run dev:build` - Watch and recompile TypeScript
- `npm run dev:full` - Full development mode (build + run)

### Adding New Methods

To add a new MCDA method:
1. Add the route to `src/app.ts`
2. Define the request and response schemas using OpenAPI specification
3. Implement the calculation logic
4. Update the API documentation tags and descriptions

## Deployment

This application is configured for deployment on Railway and other cloud platforms:

- Automatic TypeScript compilation on install
- Environment variable support for port and host
- Production-ready configuration

## Technologies Used

- **Fastify**: High-performance web framework
- **TypeScript**: Type-safe JavaScript
- **Swagger/OpenAPI**: API documentation
- **Scalar**: Modern API reference interface
- **Node.js**: Runtime environment

## License

[Your License Here]
