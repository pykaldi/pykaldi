" Vim syntax file
" Language: Clif
" Maintainer: Victor Martinez
" Latest Revision: 09 Sept 2017

if exists("b:current_syntax")
    finish
endif

" Keywords {{{
" ============
    syn match clifBrackets "{[(|)]}" contained skipwhite 

    " Namespace
    syn keyword clifStatement namespace nextgroup=clifNamespace skipwhite
    syn region clifNamespace start=+`+ end=+`+ end=+$+ keepend

    " Class definition
    syn keyword clifStatement class nextgroup=clifClass skipwhite
    syn match clifClass "\%(\%(class\s\)\s*\)\@<=\h\w*" contained nextgroup=clifClassVars
    syn region clifClassVars start="(" end=")" contained contains=clifClassParameters transparent keepend
    syn match clifClassParameters "[^,\*]*" contained contains=clifBuiltinObj,clifBuiltinTypes skipwhite

    " Clif renaming of C classes
    syn match clifCRename "\%(\%(as\s\)\s*\)\@<=\h\w*" nextgroup=clifFunctionVars

    " Function definition
    syn keyword clifStatement def nextgroup=clifFunction skipwhite
    syn match clifFunction "\%(\%(def\s\)\s*\)\@<=\h\w*" contained nextgroup=clifFunctionVars
    syn region clifFunctionVars start="(" end=")" contained contains=clifFunctionParameters transparent keepend
    syn match clifFunctionParameters "[^,]*" contained contains=clifParam skipwhite nextgroup=clifReturnValues
    syn match clifParam "[^,]*" contained contains=clifNamedParam,clifBuiltinObj,clifBuiltinTypes,clifComment,clifCRename,clifSelf,clifBuiltinDefault
    syn match clifNamedParam "\%(\h\w*\s*:\s*\)\@<=\h\w*" contained

    " Enum definition
    syn match clifEnum "\%(\%(enum\s\)\s*\)\@<=\h\w*"

" }}}

" Decorators {{{
" ==============

    syn match clifDecorator "@" display nextgroup=clifName skipwhite
    syn match clifName "\h\w*" display contained

" }}}


" Comments {{{
" ============
syn match clifComment "#.*$" display contains=clifTodo
syn keyword clifTodo TODO FIXME XXX contained


" }}}

" Strings {{{
" ===========
    syn region clifString start=+'+ skip=+\\\\\|\\'\|\\$+ excludenl end=+'+ end=+$+ keepend contains=clifEscape,clifEscapeError
    syn region clifString start=+"+ skip=+\\\\\|\\'\|\\$+ excludenl end=+"+ end=+$+ keepend contains=clifEscape,clifEscapeError
    syn region clifString start=+"""+ end=+"""+ keepend contains=clifEscape,clifEscapeError

    syn match  clifEscape         +\\[abfnrtv'"\\]+ display contained
    syn match  clifEscape         "\\\o\o\=\o\=" display contained
    syn match  clifEscapeError    "\\\o\{,2}[89]" display contained
    syn match  clifEscape         "\\x\x\{2}" display contained
    syn match  clifEscapeError    "\\x\x\=\X" display contained
    syn match clifEscape          "\\$"

    syn region clifDocstring start=+^\s*"""+ end=+"""+ keepend excludenl contains=clifEscape,clifEscapeError

" }}}

" Builtins {{{
" ============
    syn keyword clifInclude from import
    syn keyword clifStatement pass as with
    syn keyword clifBuiltinObj enum const
    syn keyword clifBuiltinTypes int bytes str bool float list tuple set dict object
    syn keyword clifBuiltinFunc property
    syn keyword clifSelf self cls
    syn keyword clifBuiltinDefault default

" }}}

" Highlight {{{
" =============
    hi def link clifInclude Include
    hi def link clifStatement Statement
    hi def link clifBuiltinObj Structure
    hi def link clifBuiltinTypes Type 
    hi def link clifBuiltinFunc Function
    hi def link clifBuiltinDefault Identifier
    hi def link clifSelf Identifier

    hi def link clifString String
    hi def link clifDocstring String

    hi def link clifEscape Special
    hi def link clifEscapeError Error

    hi def link clifComment Comment
    hi def link clifTodo Todo

    hi def link clifNamespace Structure

    hi def link clifClass Function
    hi def link clifClassParameters Normal
    hi def link clifCRename Function

    hi def link clifFunction Function
    hi def link clifFunctionVars Normal
    hi def link clifFunctionParameters Normal
    hi def link clifParam Normal
    hi def link clifNamedParam Structure
    hi def link clifBrackets Normal

    hi def link clifDecorator Define
    hi def link clifName Function
    hi def link clifEnum Function

" }}}