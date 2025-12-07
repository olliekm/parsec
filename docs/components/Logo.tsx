import React from 'react'
import ParsecLogo from '../parsec_logo.svg'
import Image from 'next/image'

type Props = {}

function Logo({}: Props) {
  return (
    <div className='flex'>
        <span className=' font-light text-xl self-center'>parsec</span>
    </div>
  )
}

export default Logo