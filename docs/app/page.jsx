import Link from 'next/link'

export default function Home() {
  return (
    <div className="relative min-h-screen bg-[#f4f3ee] text-[#463f3a]">
      {/* Header */}
      <header className="fixed top-0 left-0 z-50">
        <div className="px-6 py-6">
          <Link href="/" className="text-xl font-light text-[#463f3a] hover:text-[#8a817c] transition-colors">
            parsec
          </Link>
        </div>
      </header>

      <div className="flex flex-col items-start justify-center min-h-screen px-6 py-16 pt-24 max-w-5xl mx-auto">
        {/* Hero Section */}
        <div className="w-full mb-20">
          <div className="inline-block px-3 py-1 mb-6 text-xs font-medium text-[#463f3a] bg-[#bcb8b1]/20 rounded-full border border-[#8a817c]">
            v0.2.1 — Analytics, A/B Testing & Resilience
          </div>

          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-light mb-6 tracking-tight text-[#463f3a] leading-tight">
            The AI orchestration toolkit you can <span className="italic font-normal">trust</span>
          </h1>

          <p className="text-lg sm:text-xl text-[#8a817c] font-light mb-10 max-w-2xl leading-relaxed">
            Build production LLM applications with guaranteed structured output, intelligent failover, and data-driven optimization.
          </p>

          <div className="flex flex-row gap-3 mb-8">
            <Link
              href="/docs/get-started"
              className="px-6 py-2.5 text-sm font-medium bg-[#463f3a] text-[#f4f3ee] rounded-sm hover:bg-[#463f3a]/90 transition-all duration-200"
            >
              Get Started
            </Link>
            <Link
              href="/docs"
              className="px-6 py-2.5 text-sm font-medium bg-[#f4f3ee] text-[#463f3a] border border-[#8a817c] rounded-sm hover:border-[#463f3a] transition-colors duration-200"
            >
              Documentation
            </Link>
          </div>

          <div className="flex gap-4 text-xs text-[#8a817c]">
            <span>MIT License</span>
            <span>•</span>
            <span>Python 3.9+</span>
            <span>•</span>
            <span>OpenAI, Anthropic, Gemini, Ollama</span>
          </div>
        </div>

        {/* Code Example */}
        <div className="max-w-4xl w-full mb-24">
          <div className="bg-[#bcb8b1]/10 rounded-sm border border-[#8a817c]/30 overflow-hidden">
            <div className="p-8">
              <pre className="text-sm overflow-x-auto leading-relaxed">
                <code className="font-mono">
                  <span className="text-[#8a817c] font-semibold">from</span> <span className="text-[#463f3a]">parsec</span> <span className="text-[#8a817c] font-semibold">import</span> <span className="text-[#463f3a] font-medium">EnforcementEngine</span>{'\n'}
                  <span className="text-[#8a817c] font-semibold">from</span> <span className="text-[#463f3a]">parsec.models.adapters</span> <span className="text-[#8a817c] font-semibold">import</span> <span className="text-[#463f3a] font-medium">OpenAIAdapter</span>{'\n'}
                  <span className="text-[#8a817c] font-semibold">from</span> <span className="text-[#463f3a]">parsec.validators</span> <span className="text-[#8a817c] font-semibold">import</span> <span className="text-[#463f3a] font-medium">PydanticValidator</span>{'\n'}
                  <span className="text-[#8a817c] font-semibold">from</span> <span className="text-[#463f3a]">pydantic</span> <span className="text-[#8a817c] font-semibold">import</span> <span className="text-[#463f3a] font-medium">BaseModel</span>{'\n'}
                  {'\n'}
                  <span className="text-[#8a817c] font-semibold">class</span> <span className="text-[#463f3a] font-medium">User</span>(<span className="text-[#463f3a] font-medium">BaseModel</span>):{'\n'}
                  {'    '}<span className="text-[#463f3a]">name</span>: <span className="text-[#463f3a] font-medium">str</span>{'\n'}
                  {'    '}<span className="text-[#463f3a]">email</span>: <span className="text-[#463f3a] font-medium">str</span>{'\n'}
                  {'    '}<span className="text-[#463f3a]">age</span>: <span className="text-[#463f3a] font-medium">int</span>{'\n'}
                  {'\n'}
                  <span className="text-[#8a817c]/70"># Setup with automatic validation</span>{'\n'}
                  <span className="text-[#463f3a]">adapter</span> = <span className="text-[#463f3a] font-medium">OpenAIAdapter</span>(api_key=<span className="text-[#8a817c] font-medium">"..."</span>, model=<span className="text-[#8a817c] font-medium">"gpt-4o-mini"</span>){'\n'}
                  <span className="text-[#463f3a]">validator</span> = <span className="text-[#463f3a] font-medium">PydanticValidator</span>(){'\n'}
                  <span className="text-[#463f3a]">engine</span> = <span className="text-[#463f3a] font-medium">EnforcementEngine</span>(adapter, validator){'\n'}
                  {'\n'}
                  <span className="text-[#8a817c]/70"># Get guaranteed valid output</span>{'\n'}
                  <span className="text-[#463f3a]">result</span> = <span className="text-[#8a817c] font-semibold">await</span> engine.<span className="text-[#463f3a] font-medium">enforce</span>({'\n'}
                  {'    '}<span className="text-[#8a817c] font-medium">"Extract: John Doe, john@example.com, 30 years old"</span>,{'\n'}
                  {'    '}<span className="text-[#463f3a] font-medium">User</span>{'\n'}
                  ){'\n'}
                  {'\n'}
                  <span className="text-[#463f3a] font-medium">print</span>(result.data){'\n'}
                  <span className="text-[#8a817c]/70"># User(name='John Doe', email='john@example.com', age=30)</span>
                </code>
              </pre>
            </div>
          </div>
        </div>

        {/* Key Features Grid */}
        <div className="w-full mb-24">
          <h2 className="text-3xl font-light mb-3 text-[#463f3a]">
            Production-grade from day one
          </h2>
          <p className="text-[#8a817c] mb-12 text-base max-w-2xl">
            Everything you need to build reliable LLM applications
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="group">
              <h3 className="text-base font-medium mb-2 text-[#463f3a]">Guaranteed Structure</h3>
              <p className="text-[#8a817c] leading-relaxed text-sm">
                Automatic validation and repair with JSON Schema and Pydantic. Never parse malformed JSON again.
              </p>
            </div>

            <div className="group">
              <h3 className="text-base font-medium mb-2 text-[#463f3a]">Built-in Resilience</h3>
              <p className="text-[#8a817c] leading-relaxed text-sm">
                Circuit breakers, retry policies, and automatic failover keep your apps running when LLMs fail.
              </p>
            </div>

            <div className="group">
              <h3 className="text-base font-medium mb-2 text-[#463f3a]">Data-Driven Optimization</h3>
              <p className="text-[#8a817c] leading-relaxed text-sm">
                Track performance metrics and A/B test prompts with statistical significance testing built-in.
              </p>
            </div>

            <div className="group">
              <h3 className="text-base font-medium mb-2 text-[#463f3a]">Provider Agnostic</h3>
              <p className="text-[#8a817c] leading-relaxed text-sm">
                One API for OpenAI, Anthropic, Gemini, and Ollama. Switch providers or run A/B tests without code changes.
              </p>
            </div>

            <div className="group">
              <h3 className="text-base font-medium mb-2 text-[#463f3a]">Version-Controlled Prompts</h3>
              <p className="text-[#8a817c] leading-relaxed text-sm">
                Manage prompts as code with semantic versioning, YAML persistence, and type-safe variables.
              </p>
            </div>

            <div className="group">
              <h3 className="text-base font-medium mb-2 text-[#463f3a]">Intelligent Caching</h3>
              <p className="text-[#8a817c] leading-relaxed text-sm">
                LRU caching with TTL reduces API costs and improves response times. Monitor with built-in analytics.
              </p>
            </div>
          </div>
        </div>

        {/* Use Cases */}
        <div className="w-full mb-24">
          <h2 className="text-3xl font-light mb-12 text-[#463f3a]">
            Built for real-world applications
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border-l-2 border-[#463f3a] pl-4">
              <h4 className="font-medium text-[#463f3a] mb-1 text-sm">Data Extraction at Scale</h4>
              <p className="text-[#8a817c] text-xs leading-relaxed">
                Extract structured data from documents, emails, and unstructured text with guaranteed schema compliance.
              </p>
            </div>

            <div className="border-l-2 border-[#463f3a] pl-4">
              <h4 className="font-medium text-[#463f3a] mb-1 text-sm">API Response Generation</h4>
              <p className="text-[#8a817c] text-xs leading-relaxed">
                Ensure LLM-powered endpoints always return valid JSON. Automatic validation prevents malformed responses.
              </p>
            </div>

            <div className="border-l-2 border-[#463f3a] pl-4">
              <h4 className="font-medium text-[#463f3a] mb-1 text-sm">Multi-Step Workflows</h4>
              <p className="text-[#8a817c] text-xs leading-relaxed">
                Chain LLM calls with confidence. Validated outputs become reliable inputs, enabling complex agent workflows.
              </p>
            </div>

            <div className="border-l-2 border-[#463f3a] pl-4">
              <h4 className="font-medium text-[#463f3a] mb-1 text-sm">Classification & Routing</h4>
              <p className="text-[#8a817c] text-xs leading-relaxed">
                Get predictable categories and labels for routing logic. A/B test prompts to maximize accuracy.
              </p>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="w-full mb-24">
          <div className="grid grid-cols-3 gap-6">
            <div>
              <div className="text-3xl font-light text-[#463f3a] mb-1">4</div>
              <div className="text-xs text-[#8a817c]">LLM Providers</div>
            </div>
            <div>
              <div className="text-3xl font-light text-[#463f3a] mb-1">100%</div>
              <div className="text-xs text-[#8a817c]">Schema Compliance</div>
            </div>
            <div>
              <div className="text-3xl font-light text-[#463f3a] mb-1">0.2.1</div>
              <div className="text-xs text-[#8a817c]">Latest Version</div>
            </div>
          </div>
        </div>

        {/* Final CTA */}
        <div className="w-full mb-16">
          <h2 className="text-2xl font-light mb-4 text-[#463f3a]">
            Stop fighting with LLM outputs
          </h2>
          <p className="text-base text-[#8a817c] mb-6 leading-relaxed max-w-2xl">
            Install with pip and start building reliable LLM applications in minutes.
          </p>
          <div className="bg-[#463f3a] rounded-sm p-4 mb-6 inline-block">
            <code className="text-[#ffffff] font-mono text-sm px-4">pip install parsec-llm</code>
          </div>
          <div className="flex flex-row  gap-3">
            <Link
              href="/docs/get-started"
              className="px-6 py-2.5 text-sm font-medium bg-[#463f3a] text-[#f4f3ee] rounded-sm hover:bg-[#463f3a]/90 transition-all duration-200 inline-block"
            >
              Get Started
            </Link>
            <a
              href="https://github.com/olliekm/parsec"
              className="px-6 py-2.5 text-sm font-medium bg-[#f4f3ee] text-[#463f3a] border border-[#8a817c] rounded-sm hover:border-[#463f3a] transition-colors duration-200 inline-block"
              target="_blank"
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </div>
        </div>

        {/* Footer Links */}
        <div className="w-full border-t border-[#8a817c]/30 pt-8">
          <div className="flex gap-6 text-xs mb-3">
            <a
              href="https://github.com/olliekm/parsec"
              className="text-[#8a817c] hover:text-[#463f3a] transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
            <a
              href="https://github.com/olliekm/parsec/issues"
              className="text-[#8a817c] hover:text-[#463f3a] transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Issues
            </a>
            <a
              href="https://pypi.org/project/parsec-llm/"
              className="text-[#8a817c] hover:text-[#463f3a] transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              PyPI
            </a>
            <Link
              href="/docs"
              className="text-[#8a817c] hover:text-[#463f3a] transition-colors"
            >
              Documentation
            </Link>
          </div>
          <p className="text-xs text-[#8a817c]">
            MIT License • Created by <a href="https://olliekm.com" className="hover:text-[#463f3a] transition-colors" target="_blank" rel="noopener noreferrer">Oliver Kwun-Morfitt</a>
          </p>
        </div>
      </div>
    </div>
  )
}
