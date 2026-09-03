import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { scanConformancePolicy } from './conformance-policy.mjs'

const CONFIG_NAME = 'design.config.json'
const LOCK_NAME = 'design.lock.json'
const MANAGED_START = '/* kafka-design:managed-start */'
const MANAGED_END = '/* kafka-design:managed-end */'
const SHA40 = /^[0-9a-f]{40}$/
const SHA64 = /^[0-9a-f]{64}$/
const CONFIG_KEYS = new Set(['$schema', 'schemaVersion', 'designSha', 'preset', 'cssEntry', 'managedDir', 'logo'])
const LOCK_KEYS = new Set(['schemaVersion', 'designSha', 'preset', 'configHash', 'manifestHash', 'logoHash', 'installedRegistryItems', 'managedFiles', 'integration'])
const MANAGED_SOURCES = Object.freeze([
  { source: 'styles/tokens.css', target: 'kafka-tokens.css', css: true },
  { source: 'styles/globals.css', target: 'kafka-globals.css', css: true },
  { source: 'styles/components.css', target: 'kafka-components.css', css: true },
  { source: 'scripts/portable-conformance.mjs', target: 'portable-conformance.mjs', css: false },
  { source: 'scripts/conformance-policy.mjs', target: 'conformance-policy.mjs', css: false },
])

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function stableStringify(value) {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function toPosix(relativePath) {
  return relativePath.split(path.sep).join('/')
}

function push(errors, rule, filePath, message) {
  errors.push({ rule, path: toPosix(filePath), message })
}

function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be a JSON object`)
}

function assertExactKeys(value, allowed, label) {
  for (const key of Object.keys(value)) if (!allowed.has(key)) throw new Error(`${label} has unknown field: ${key}`)
}

function readJson(filePath, label) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'))
  } catch (error) {
    throw new Error(`${label} is missing or invalid JSON: ${filePath}\n${error.message}`)
  }
}

function resolveWithin(root, relativePath, label) {
  if (typeof relativePath !== 'string' || relativePath.length === 0) throw new Error(`${label} must be a non-empty relative path`)
  if (path.isAbsolute(relativePath)) throw new Error(`${label} must be relative: ${relativePath}`)
  const resolvedRoot = path.resolve(root)
  const resolved = path.resolve(resolvedRoot, relativePath)
  if (resolved !== resolvedRoot && !resolved.startsWith(`${resolvedRoot}${path.sep}`)) throw new Error(`${label} escapes consumer root: ${relativePath}`)
  return resolved
}

function validateConfig(config) {
  assertPlainObject(config, CONFIG_NAME)
  assertExactKeys(config, CONFIG_KEYS, CONFIG_NAME)
  if (config.schemaVersion !== 1) throw new Error(`${CONFIG_NAME}.schemaVersion must be 1`)
  if (!SHA40.test(config.designSha ?? '')) throw new Error(`${CONFIG_NAME}.designSha must be a full 40-character lowercase Git SHA`)
  if (!['base', 'financial-dashboard'].includes(config.preset)) throw new Error(`${CONFIG_NAME}.preset must be base or financial-dashboard`)
  for (const key of ['cssEntry', 'managedDir']) if (typeof config[key] !== 'string' || config[key].length === 0) throw new Error(`${CONFIG_NAME}.${key} must be a non-empty string`)
  if (config.logo !== undefined && (typeof config.logo !== 'string' || config.logo.length === 0)) throw new Error(`${CONFIG_NAME}.logo must be a non-empty string when present`)
  return config
}

function validateLock(lock) {
  assertPlainObject(lock, LOCK_NAME)
  assertExactKeys(lock, LOCK_KEYS, LOCK_NAME)
  if (lock.schemaVersion !== 1) throw new Error(`${LOCK_NAME}.schemaVersion must be 1`)
  if (!SHA40.test(lock.designSha ?? '')) throw new Error(`${LOCK_NAME}.designSha must be a full Git SHA`)
  if (!['base', 'financial-dashboard'].includes(lock.preset)) throw new Error(`${LOCK_NAME}.preset is invalid`)
  for (const key of ['configHash', 'manifestHash']) if (!SHA64.test(lock[key] ?? '')) throw new Error(`${LOCK_NAME}.${key} must be sha256`)
  if (lock.logoHash !== null && !SHA64.test(lock.logoHash ?? '')) throw new Error(`${LOCK_NAME}.logoHash must be sha256 or null`)
  if (!Array.isArray(lock.installedRegistryItems) || lock.installedRegistryItems.some((item) => typeof item !== 'string')) throw new Error(`${LOCK_NAME}.installedRegistryItems must be a string array`)
  if (!Array.isArray(lock.managedFiles)) throw new Error(`${LOCK_NAME}.managedFiles must be an array`)
  for (const item of lock.managedFiles) {
    assertPlainObject(item, `${LOCK_NAME}.managedFiles[]`)
    assertExactKeys(item, new Set(['path', 'sha256']), `${LOCK_NAME}.managedFiles[]`)
    if (typeof item.path !== 'string' || item.path.length === 0 || !SHA64.test(item.sha256 ?? '')) throw new Error(`${LOCK_NAME}.managedFiles[] has invalid path or sha256`)
  }
  assertPlainObject(lock.integration, `${LOCK_NAME}.integration`)
  assertExactKeys(lock.integration, new Set(['cssEntry', 'blockHash']), `${LOCK_NAME}.integration`)
  if (typeof lock.integration.cssEntry !== 'string' || lock.integration.cssEntry.length === 0 || !SHA64.test(lock.integration.blockHash ?? '')) throw new Error(`${LOCK_NAME}.integration is invalid`)
  return lock
}

function expectedManagedPaths(config) {
  return MANAGED_SOURCES.map((item) => ({
    ...item,
    consumerPath: toPosix(path.join(config.managedDir, item.target)),
  }))
}

function buildIntegration(consumerRoot, config, managed) {
  const cssEntryPath = resolveWithin(consumerRoot, config.cssEntry, `${CONFIG_NAME}.cssEntry`)
  const cssDir = path.dirname(cssEntryPath)
  const imports = managed.filter((item) => item.css).map((item) => {
    const targetPath = resolveWithin(consumerRoot, item.consumerPath, `managed target ${item.target}`)
    let relative = toPosix(path.relative(cssDir, targetPath))
    if (!relative.startsWith('.')) relative = `./${relative}`
    return `@import "${relative}";`
  })
  const block = `${MANAGED_START}\n${imports.join('\n')}\n${MANAGED_END}`
  return { cssEntryPath, block, blockHash: sha256(block) }
}

function expectedLogoHash(consumerRoot, config) {
  if (config.logo === undefined) return null
  const logoPath = resolveWithin(consumerRoot, config.logo, `${CONFIG_NAME}.logo`)
  if (!fs.existsSync(logoPath) || !fs.statSync(logoPath).isFile()) throw new Error(`Configured logo is missing: ${config.logo}`)
  return sha256(fs.readFileSync(logoPath))
}

export function checkPortableConsumer(consumerPath) {
  const consumerRoot = path.resolve(consumerPath)
  if (!fs.existsSync(consumerRoot) || !fs.statSync(consumerRoot).isDirectory()) throw new Error(`Consumer directory does not exist: ${consumerRoot}`)

  const config = validateConfig(readJson(path.join(consumerRoot, CONFIG_NAME), CONFIG_NAME))
  resolveWithin(consumerRoot, config.cssEntry, `${CONFIG_NAME}.cssEntry`)
  resolveWithin(consumerRoot, config.managedDir, `${CONFIG_NAME}.managedDir`)
  if (config.logo !== undefined) resolveWithin(consumerRoot, config.logo, `${CONFIG_NAME}.logo`)

  const errors = []
  const lockPath = path.join(consumerRoot, LOCK_NAME)
  let lock
  try {
    lock = validateLock(readJson(lockPath, LOCK_NAME))
  } catch (error) {
    push(errors, 'managed-file-drift', LOCK_NAME, error.message)
    return [...errors, ...scanConformancePolicy(consumerRoot, config)]
  }

  if (lock.designSha !== config.designSha) push(errors, 'managed-file-drift', LOCK_NAME, 'design SHA does not match design.config.json')
  if (lock.preset !== config.preset) push(errors, 'managed-file-drift', LOCK_NAME, 'preset does not match design.config.json')
  if (lock.configHash !== sha256(stableStringify(config))) push(errors, 'managed-file-drift', LOCK_NAME, 'configHash does not match design.config.json')
  if (lock.logoHash !== expectedLogoHash(consumerRoot, config)) push(errors, 'managed-file-drift', LOCK_NAME, 'logoHash does not match configured logo')
  if (lock.installedRegistryItems.length !== 0) push(errors, 'managed-file-drift', LOCK_NAME, 'portable CSS adoption expects installedRegistryItems to be empty')

  const expectedManaged = expectedManagedPaths(config)
  const recorded = new Map(lock.managedFiles.map((item) => [toPosix(item.path), item.sha256]))
  if (recorded.size !== lock.managedFiles.length) push(errors, 'managed-file-drift', LOCK_NAME, 'managedFiles contains duplicate paths')
  const expectedPaths = new Set(expectedManaged.map((item) => item.consumerPath))
  for (const recordedPath of recorded.keys()) if (!expectedPaths.has(recordedPath)) push(errors, 'managed-file-drift', LOCK_NAME, `unexpected managed file recorded: ${recordedPath}`)

  const manifestAssets = []
  for (const item of expectedManaged) {
    const recordedHash = recorded.get(item.consumerPath)
    if (!recordedHash) {
      push(errors, 'managed-file-drift', item.consumerPath, 'managed file is missing from design.lock.json')
      continue
    }
    const filePath = resolveWithin(consumerRoot, item.consumerPath, `managed file ${item.consumerPath}`)
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      push(errors, 'managed-file-drift', item.consumerPath, 'managed file is missing')
    } else if (sha256(fs.readFileSync(filePath)) !== recordedHash) {
      push(errors, 'managed-file-drift', item.consumerPath, 'managed file was edited or is stale')
    }
    manifestAssets.push({ source: item.source, target: item.consumerPath, sha256: recordedHash })
  }

  if (manifestAssets.length === expectedManaged.length) {
    const manifest = { schemaVersion: 1, preset: config.preset, assets: manifestAssets }
    if (lock.manifestHash !== sha256(stableStringify(manifest))) push(errors, 'managed-file-drift', LOCK_NAME, 'manifestHash does not match managed files')
  }

  const integration = buildIntegration(consumerRoot, config, expectedManaged)
  if (lock.integration.cssEntry !== toPosix(config.cssEntry) || lock.integration.blockHash !== integration.blockHash) {
    push(errors, 'managed-file-drift', LOCK_NAME, 'integration does not match canonical CSS import block')
  }
  if (!fs.existsSync(integration.cssEntryPath)) {
    push(errors, 'managed-file-drift', config.cssEntry, 'configured CSS entry is missing')
  } else {
    const cssEntry = fs.readFileSync(integration.cssEntryPath, 'utf8')
    const starts = cssEntry.split(MANAGED_START).length - 1
    const ends = cssEntry.split(MANAGED_END).length - 1
    if (starts !== 1 || ends !== 1 || !cssEntry.includes(integration.block)) push(errors, 'managed-file-drift', config.cssEntry, 'canonical managed import block is missing, duplicated, or edited')
  }

  return [...errors, ...scanConformancePolicy(consumerRoot, config)]
}

function parseConsumerArg(argv) {
  const index = argv.indexOf('--consumer')
  if (index === -1 || !argv[index + 1] || argv[index + 1].startsWith('--')) throw new Error(`Usage: node portable-conformance.mjs --consumer <path containing ${CONFIG_NAME}>`)
  if (argv.length !== index + 2) throw new Error('Unexpected conformance arguments')
  return argv[index + 1]
}

const isCli = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isCli) {
  try {
    const errors = checkPortableConsumer(parseConsumerArg(process.argv.slice(2)))
    if (errors.length === 0) console.log('design conformance: ok')
    else {
      for (const error of errors) console.error(`[${error.rule}] ${error.path}: ${error.message}`)
      process.exitCode = 1
    }
  } catch (error) {
    console.error(`design conformance failed: ${error.message}`)
    process.exitCode = 1
  }
}
