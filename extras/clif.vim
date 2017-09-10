" Vim syntax file
" Language: Clif
" Maintainer: Victor Martinez
" Latest Revision: 09 Sept 2017

if exists("b:current_syntax")
    finish
endif

" Keywords {{{
" ============

" Namespace
    syn keyword clifStatement namespace nextgroup=clifNamespace skipwhite
    syn region clifNamespace start=+`+ end=+`+ end=+$+ keepend

" Class definition
    syn keyword clifStatement class nextgroup=clifClass skipwhite

    syn match clifClass "\%(\%(class\s\)\s*\)\@<=\h\w*" contained nextgroup=clifClassVars
    syn region clifClassVars start=+(+ end=+)+ contained contains=clifClassParameters transparent keepend
    syn match clifClassParameters "[^,\*]*" contained contains=clifBuiltinObj skipwhite

    syn match clifCRename "\%(\%(as\s\)\s*\)\@<=\h\w*"
    
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
    syn keyword clifStatement pass as
    syn keyword clifBuiltinObj enum const
    syn keyword clifBuiltinTypes int bytes str bool float list tuple set dict object
    syn keyword clifBuiltinFunc property

" }}}

" Highlight {{{
" =============
    hi def link clifInclude Include
    hi def link clifStatement Statement
    hi def link clifBuiltinObj Structure
    hi def link clifBuiltinTypes Type 
    hi def link clifBuiltinFunc Function

    hi def link clifString String
    hi def link clifDocstring String

    hi def link clifEscape Special
    hi def link clifEscapeError Error

    hi def link clifComment Comment
    hi def link clifTodo Todo

    hi def link clifNamespace Structure

    hi def link clifClass Function
    hi def link clifClassRename Structure 
    hi def link clifClassParameters Normal
    hi def link clifCRename Function
" }}}