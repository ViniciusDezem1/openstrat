import Fastify, { FastifyInstance } from 'fastify'
import FastifySwagger from '@fastify/swagger'
import FastifySwaggerUi from '@fastify/swagger-ui'

// Instantiate the framework
const fastify: FastifyInstance = Fastify({
  logger: true,
})

// Set up @fastify/swagger
await fastify.register(FastifySwagger, {
  openapi: {
    info: {
      title: 'Open Strat: Open Finance Decision Support API',
      version: '1.0.0',
      description: 'A comprehensive API for Multi-Criteria Decision Analysis (MCDA) methods supporting financial decision-making processes. Includes Fuzzy Analytic Hierarchy Process (FAHP), Optimization algorithms, and Strategic Value Index (SVI) implementations.'
    },
    components: {
      securitySchemes: {
        apiKey: {
          type: 'apiKey',
          name: 'apiKey',
          in: 'header',
        },
      },
    },
  },
})

// Set up Swagger UI
await fastify.register(FastifySwaggerUi, {
  routePrefix: '/documentation',
  uiConfig: {
    docExpansion: 'full',
    deepLinking: false,
  },
  uiHooks: {
    onRequest: function (request, reply, next) {
      next()
    },
    preHandler: function (request, reply, next) {
      next()
    },
  },
  staticCSP: true,
  transformStaticCSP: (header) => header,
  transformSpecification: (swaggerObject, request, reply) => {
    return swaggerObject
  },
  transformSpecificationClone: true,
})

// Set up Scalar API Reference
await fastify.register(import('@scalar/fastify-api-reference'), {
  routePrefix: '/reference',
  configuration: {
    title: 'Open Finance Decision Support API Reference',
    theme: 'purple',
  },
})

// Root redirect to reference
fastify.get('/', async function handler (request, reply) {
  return reply.redirect('/reference')
})

// FAHP API endpoints
fastify.get('/fahp', {
  schema: {
    description: 'Get Fuzzy Analytic Hierarchy Process (FAHP) data and analysis results',
    tags: ['fahp'],
    summary: 'Retrieve FAHP analysis information and status',
    response: {
      200: {
        description: 'Successful response',
        type: 'object',
        properties: {
          message: { type: 'string' },
          data: { type: 'object' }
        },
      },
    },
  },
}, async function handler (request, reply) {
  return { message: 'FAHP API', data: { status: 'active' } }
})

fastify.post('/fahp', {
  schema: {
    description: 'Create new Fuzzy Analytic Hierarchy Process (FAHP) analysis for multi-criteria decision making',
    tags: ['fahp'],
    summary: 'Create new FAHP analysis with criteria and fuzzy values',
    body: {
      type: 'object',
      properties: {
        name: { type: 'string' },
        value: { type: 'number' }
      },
      required: ['name', 'value']
    },
    response: {
      201: {
        description: 'Created successfully',
        type: 'object',
        properties: {
          message: { type: 'string' },
          id: { type: 'string' }
        },
      },
    },
  },
}, async function handler (request, reply) {
  const body = request.body as { name: string; value: number }
  reply.code(201)
  return { message: 'FAHP created', id: 'fahp_' + Date.now() }
})

// Optimization API endpoints
fastify.get('/optimization', {
  schema: {
    description: 'Get multi-objective optimization data and algorithm status',
    tags: ['optimization'],
    summary: 'Retrieve optimization algorithms and results',
    response: {
      200: {
        description: 'Successful response',
        type: 'object',
        properties: {
          message: { type: 'string' },
          data: { type: 'object' }
        },
      },
    },
  },
}, async function handler (request, reply) {
  return { message: 'Optimization API', data: { status: 'active' } }
})

fastify.post('/optimization', {
  schema: {
    description: 'Execute multi-objective optimization algorithms for financial decision support',
    tags: ['optimization'],
    summary: 'Run optimization algorithms with custom parameters',
    body: {
      type: 'object',
      properties: {
        algorithm: { type: 'string' },
        parameters: { type: 'object' }
      },
      required: ['algorithm']
    },
    response: {
      201: {
        description: 'Optimization started',
        type: 'object',
        properties: {
          message: { type: 'string' },
          jobId: { type: 'string' }
        },
      },
    },
  },
}, async function handler (request, reply) {
  const body = request.body as { algorithm: string; parameters?: object }
  reply.code(201)
  return { message: 'Optimization started', jobId: 'opt_' + Date.now() }
})

// SVI API endpoints
fastify.get('/svi', {
  schema: {
    description: 'Get Strategic Value Index (SVI) data and assessment results',
    tags: ['svi'],
    summary: 'Retrieve SVI analysis and strategic value metrics',
    response: {
      200: {
        description: 'Successful response',
        type: 'object',
        properties: {
          message: { type: 'string' },
          data: { type: 'object' }
        },
      },
    },
  },
}, async function handler (request, reply) {
  return { message: 'SVI API', data: { status: 'active' } }
})

fastify.post('/svi', {
  schema: {
    description: 'Create new Strategic Value Index (SVI) entry for investment opportunity assessment',
    tags: ['svi'],
    summary: 'Create new SVI analysis with index values and strategic metrics',
    body: {
      type: 'object',
      properties: {
        index: { type: 'string' },
        value: { type: 'number' },
        date: { type: 'string', format: 'date' }
      },
      required: ['index', 'value']
    },
    response: {
      201: {
        description: 'Created successfully',
        type: 'object',
        properties: {
          message: { type: 'string' },
          id: { type: 'string' }
        },
      },
    },
  },
}, async function handler (request, reply) {
  const body = request.body as { index: string; value: number; date?: string }
  reply.code(201)
  return { message: 'SVI created', id: 'svi_' + Date.now() }
})

// Serve an OpenAPI file
fastify.get('/openapi.json', async (request, reply) => {
  return fastify.swagger()
})

// Wait for Fastify
await fastify.ready()

// Run the server
const start = async () => {
  try {
    const port = process.env.PORT || 3000
    const host = process.env.HOST || '0.0.0.0'

    await fastify.listen({ port: Number(port), host })
    console.log(`Fastify is now listening on http://${host}:${port}`)
    console.log(`Swagger UI available at http://${host}:${port}/documentation`)
    console.log(`Scalar API Reference available at http://${host}:${port}/reference`)
  } catch (err) {
    fastify.log.error(err)
    process.exit(1)
  }
}

start()
