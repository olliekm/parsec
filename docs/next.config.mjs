import nextra from 'nextra'

const withNextra = nextra({
  search: {
    codeblocks: true,
  },
  defaultShowCopyCode: true
})

export default withNextra({
  pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],
  output: 'export',
  images: {
    unoptimized: true
  },
  turbopack: {
    resolveAlias: {
      'next-mdx-import-source-file': './mdx-components.jsx'
    }
  }

})
