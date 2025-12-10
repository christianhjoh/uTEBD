using Test
using uTEBD


@testset "uTEBD basic tests" begin
    @test isdefined(uTEBD, :iTEBDstep)
end